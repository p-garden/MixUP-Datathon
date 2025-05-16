import os
import time
import pandas as pd
from tqdm import tqdm
from typing import Dict, List
import asyncio
import aiohttp
import requests
from aiohttp import ClientTimeout  # 상단 import 추가
from config import ExperimentConfig
from prompts.templates import TEMPLATES
from utils.metrics import evaluate_correction
import re



class ExperimentRunner:
    def __init__(self, config: ExperimentConfig, api_key: str):
        self.config = config
        self.api_key = api_key
        self.template = TEMPLATES[config.template_name]
        self.api_url = config.api_url
        self.model = config.model
    
    def _make_prompt(self, text: str) -> str:
        """프롬프트 생성"""
        return self.template.format(text=text)
    
    async def _call_api_batch_async(self, prompts: List[str]) -> List[str]:
        """템플릿 기반 멀티턴 방식으로 각 문장에 대해 3단계 교정 수행"""
        semaphore = asyncio.Semaphore(3)  # 🔐 동시에 3개까지만 실행

        async def fetch_multi_turn(session, text, max_retries=3):
            async with semaphore:  # ✅ 세마포어 제한 적용
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }

                async def call(messages):
                    wait_time = 3
                    for attempt in range(max_retries):
                        try:
                            payload = {
                                "model": self.model,
                                "messages": messages,
                                "temperature": self.config.temperature
                            }
                            async with session.post(self.api_url, headers=headers, json=payload) as response:
                                response.raise_for_status()
                                result = await response.json()
                                return result["choices"][0]["message"]["content"]
                        except aiohttp.ClientResponseError as e:
                            if e.status == 429 and attempt < max_retries - 1:
                                await asyncio.sleep(wait_time)
                                wait_time *= 2
                            else:
                                raise e
                        except (aiohttp.ClientOSError, asyncio.TimeoutError) as e:
                            if attempt < max_retries - 1:
                                await asyncio.sleep(wait_time)
                                wait_time *= 2
                            else:
                                raise e

                # ✅ 싱글턴 메시지 구성
                messages = [
                    {"role": "system", "content": "너는 한국어 문장을 교정하는 AI야. 사용자의 문장을 보고 오류를 고쳐줘."},
                    {"role": "user", "content": self.template.format(text=text)}
                ]

                # ✅ API 호출 (단 한 번)
                response = await call(messages)


                return response

        timeout = ClientTimeout(total=30)  # 최대 30초 대기 허용
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = [fetch_multi_turn(session, prompt) for prompt in prompts]
            return await asyncio.gather(*tasks)
            
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """데이터셋에 대한 실험 실행 (비동기 배치 처리 + 중간 저장은 test셋에만 적용)"""
        results = []
        batch_size = self.config.batch_size
        use_intermediate = self.config.experiment_name == "final_submission6"
        save_path = f"outputs/intermediate_{self.config.experiment_name}.csv"
        def clean_output(err: str, cor: str) -> str:
            """따옴표 조건부 제거 + 공백 제거"""
            cor = re.sub(r'[\s\u200b\u200c\u200d\ufeff]+$', '', cor)  # 끝 공백 제거
            cor = re.sub(r'\s+(?=["\'])', '', cor)  # 따옴표 앞 공백 제거

            # 입력이 따옴표로 감싸져 있지 않으면 출력에서 따옴표 제거
            if not (err.startswith('"') and err.endswith('"')) and \
            not (err.startswith("'") and err.endswith("'")):
                if (cor.startswith('"') and cor.endswith('"')) or \
                (cor.startswith("'") and cor.endswith("'")):
                    cor = cor[1:-1].strip()
            return cor 
        # ✅ 이미 처리된 id 불러오기 (재시작 지원, test셋만)
        processed_ids = set()
        if use_intermediate and os.path.exists(save_path):
            existing = pd.read_csv(save_path)
            processed_ids = set(existing['id'].tolist())
            print(f"✅ 재시작 감지: {len(processed_ids)}개 문장이 이미 처리됨.")
            results.extend(existing.to_dict(orient='records'))

        # ✅ 아직 처리되지 않은 데이터 필터링
        data = data[~data['id'].isin(processed_ids)].reset_index(drop=True)
        print(f"▶️ 남은 처리 대상: {len(data)}개 문장")

        for i in tqdm(range(0, len(data), batch_size), desc="API 호출 중"):
            batch = data.iloc[i:i + batch_size]
            prompts = [row['err_sentence'] for _, row in batch.iterrows()]

            # ✅ 비동기 처리
            responses = asyncio.run(self._call_api_batch_async(prompts))

            batch_results = []
            for (_, row), corrected in zip(batch.iterrows(), responses):
                cleaned = clean_output(row['err_sentence'], corrected)  # ✅ 후처리 적용
                result = {
                    'id': row['id'],
                    'err_sentence': row['err_sentence'],
                    'cor_sentence': cleaned
                }
                results.append(result)
                batch_results.append(result)

            # ✅ 중간 결과 저장 (test셋만)
            if use_intermediate:
                pd.DataFrame(batch_results).to_csv(
                    save_path,
                    index=False,
                    mode='a',
                    header=not os.path.exists(save_path) and i == 0
                )

        return pd.DataFrame(results)

    def run_template_experiment(self, train_data: pd.DataFrame, valid_data: pd.DataFrame) -> Dict:
        """템플릿별 실험 실행"""
        print(f"\n=== {self.config.template_name} 템플릿 실험 ===")
        
        # 학습 데이터로 실험
        print("\n[학습 데이터 실험]")
        train_results = self.run(train_data)
        train_recall = evaluate_correction(train_data, train_results)
        
        # 검증 데이터로 실험
        print("\n[검증 데이터 실험]")
        valid_results = self.run(valid_data)
        valid_recall = evaluate_correction(valid_data, valid_results)
        
        return {
            'train_recall': train_recall,
            'valid_recall': valid_recall,
            'train_results': train_results,
            'valid_results': valid_results
        } 
    def _call_api_single(self, prompt: str) -> str:
        """단일 문장에 대한 API 호출"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        response = requests.post(self.api_url, headers=headers, json=data)
        response.raise_for_status()
        results = response.json()
        return results["choices"][0]["message"]["content"]