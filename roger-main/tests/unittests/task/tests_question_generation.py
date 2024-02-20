from roger.task.question_generation import QuestionGenerationTask
from tests.sample import openai_keys


SAMPLE_QUERY = """BERT에 대해 설명해줘"""
SAMPLE_ANSWER = """BERT는 Bidirectional Encoder Representations from Transformers의 약자입니다. 이는  Google에서 개발한 자연어 처리 모델로, 2018년에 발표되었습니다. BERT는 Transformer 아키텍처를 기반으로 하며, 양방향으로 텍스트를 처리하는 능력을 갖추고 있습니다.
BERT 모델은 사전 학습과 fine-tuning 두 단계로 이루어집니다. 우선, 대규모의 텍스트 데이터로 사전 학습을 진행합니다. 이 과정에서 BERT는 밑바닥부터 언어의 문맥을 학습하고, 다수의 자연어 처리 태스크를 동시에 학습하도록 설계되었습니다. """


def test_question_generation_task(openai_keys):
    openai_api_key, openai_org_key = openai_keys

    _, questions = QuestionGenerationTask(
        question=SAMPLE_QUERY,
        answer=SAMPLE_ANSWER,
        api_key=openai_api_key,
        org_key=openai_org_key,
    )

    assert isinstance(questions, list)
