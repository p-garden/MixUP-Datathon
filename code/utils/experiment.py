import os
import time
import pandas as pd
from tqdm import tqdm
from typing import Dict, List
import asyncio
import aiohttp
import requests

from config import ExperimentConfig
from prompts.templates import TEMPLATES
from utils.metrics import evaluate_correction



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
        """비동기 방식으로 여러 문장을 병렬로 API 호출 (429 에러 대비 백오프 포함)"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        semaphore = asyncio.Semaphore(5)  # 동시에 5개까지 요청 제한

        async def fetch(session, prompt, max_retries=3):
            wait_time = 1  # 초기 대기 시간
            for attempt in range(max_retries):
                async with semaphore:
                    try:
                        data = {
                            "model": self.model,
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": self.config.temperature
                        }
                        async with session.post(self.api_url, headers=headers, json=data) as response:
                            response.raise_for_status()
                            result = await response.json()
                            return result["choices"][0]["message"]["content"]
                    except aiohttp.ClientResponseError as e:
                        if e.status == 429 and attempt < max_retries - 1:
                            await asyncio.sleep(wait_time)
                            wait_time *= 2  # 지수 백오프
                        else:
                            raise e

        async with aiohttp.ClientSession() as session:
            tasks = [fetch(session, prompt) for prompt in prompts]
            return await asyncio.gather(*tasks) 
    """
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        데이터셋에 대한 실험 실행 (배치 처리)
        results = []
        batch_size = self.config.batch_size

        for i in tqdm(range(0, len(data), batch_size), desc="API 호출 중"):
            batch = data.iloc[i:i + batch_size]
            prompts = [self._make_prompt(row['err_sentence']) for _, row in batch.iterrows()]
            responses = self._call_api_batch(prompts)

            for (_, row), corrected in zip(batch.iterrows(), responses):
                results.append({
                    'id': row['id'],
                    'cor_sentence': corrected
                })

        return pd.DataFrame(results)
    """
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """데이터셋에 대한 실험 실행 (비동기 배치 처리)"""
        results = []
        batch_size = self.config.batch_size

        for i in tqdm(range(0, len(data), batch_size), desc="API 호출 중"):
            batch = data.iloc[i:i + batch_size]
            prompts = [self._make_prompt(row['err_sentence']) for _, row in batch.iterrows()]
            
            # ✅ 비동기 처리
            responses = asyncio.run(self._call_api_batch_async(prompts))

            for (_, row), corrected in zip(batch.iterrows(), responses):
                results.append({
                    'id': row['id'],
                    'cor_sentence': corrected
                })

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