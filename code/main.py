import os
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import re

from config import ExperimentConfig
from prompts.templates import TEMPLATES
from utils.experiment import ExperimentRunner

def main():
    # API 키 로드
    load_dotenv()
    api_key = os.getenv('UPSTAGE_API_KEY')
    if not api_key:
        raise ValueError("API key not found in environment variables")
    
    # 기본 설정 생성
    base_config = ExperimentConfig(template_name='basic')
    
    # 데이터 로드
    train = pd.read_csv(os.path.join(base_config.data_dir, 'train.csv'))
    test = pd.read_csv(os.path.join(base_config.data_dir, 'test.csv'))
    
    # 토이 데이터셋 생성
    toy_data = train.sample(n=base_config.toy_size, random_state=base_config.random_seed).reset_index(drop=True)
    
    # train/valid 분할
    train_data, valid_data = train_test_split(
        toy_data,
        test_size=base_config.test_size,
        random_state=base_config.random_seed
    )
    
    # 모든 템플릿으로 실험
    results = {}
    for template_name in tqdm(TEMPLATES.keys(), desc="템플릿 실험 진행", ncols=80):
        config = ExperimentConfig(
            template_name=template_name,
            temperature=0.3,
            batch_size=4,
            experiment_name=f"toy_experiment_{template_name}"
        )
        runner = ExperimentRunner(config, api_key)
        results[template_name] = runner.run_template_experiment(train_data, valid_data)
        # 🔽 🔽 여기에 추가
        train_inputs = train_data[["id", "err_sentence", "cor_sentence"]].reset_index(drop=True)
        train_inputs = train_inputs.rename(columns={"cor_sentence": "answer"})  # 일치시키기 위해
        train_outputs = runner.run(train_inputs)
        train_results = train_inputs.copy()
        train_results["cor_sentence"] = train_outputs["cor_sentence"]
        train_results.to_csv(f"outputs/train_outputs_{template_name}.csv", index=False)    
    # 결과 비교
    print("\n=== 템플릿별 성능 비교 ===")
    for template_name, result in results.items():
        print(f"\n[{template_name} 템플릿]")
        print("Train Recall:", f"{result['train_recall']['recall']:.2f}%")
        print("Train Precision:", f"{result['train_recall']['precision']:.2f}%")
        print("\nValid Recall:", f"{result['valid_recall']['recall']:.2f}%")
        print("Valid Precision:", f"{result['valid_recall']['precision']:.2f}%")
    
    # 최고 성능 템플릿 찾기
    best_template = max(
        results.items(), 
        key=lambda x: x[1]['valid_recall']['recall']
    )[0]
    
    print(f"\n최고 성능 템플릿: {best_template}")
    print(f"Valid Recall: {results[best_template]['valid_recall']['recall']:.2f}%")
    print(f"Valid Precision: {results[best_template]['valid_recall']['precision']:.2f}%")

    # 최고 성능 템플릿으로 예시 문장 몇 개 생성해서 출력
    print("\n[최고 템플릿 예시 응답]")
    sample_inputs = test["err_sentence"].head(3).tolist()

    preview_config = ExperimentConfig(
        template_name=best_template,
        temperature=0.3,
        batch_size=4,
        experiment_name="preview_generation"
    )
    preview_runner = ExperimentRunner(preview_config, api_key)

    # 임시 DataFrame 구성
    preview_df = test[["id", "err_sentence"]].head(3).reset_index(drop=True)
    preview_outputs = preview_runner.run(preview_df)

    # 결과 출력
    for err, cor in zip(sample_inputs, preview_outputs["cor_sentence"]):
        print(f"\n[입력] {err}\n[출력] {cor}")

    # 최고 성능 템플릿으로 제출 파일 생성
    print("\n=== 테스트 데이터 예측 시작 ===")
    config = ExperimentConfig(
        template_name=best_template,
        temperature=0.3,
        batch_size=4,
        experiment_name="final_submission"
    )
    
    runner = ExperimentRunner(config, api_key)
    test_results = runner.run(test)
    # 문장부호 뒤 공백 제거 함수
    def remove_trailing_space_after_punctuation(text):
        text = text.rstrip()  # 문자열 끝 전체에서 공백 제거
        return re.sub(r'([.?!…])$', r'\1', text)

    # 문장부호 뒤 공백 제거 적용
    test_results["cor_sentence"] = test_results["cor_sentence"].apply(remove_trailing_space_after_punctuation)
    # sample_submission 형식에 맞게 생성
    output = test.copy()
    output["cor_sentence"] = test_results["cor_sentence"]
    output = output[["id", "err_sentence", "cor_sentence"]]
    output.to_csv("outputs/submission_multiturn1.csv", index=False)

    print("\n제출 파일이 생성되었습니다: submission_multiturn1.csv")
    print(f"사용된 템플릿: {best_template}")
    print(f"예측된 샘플 수: {len(output)}")

if __name__ == "__main__":
    main()