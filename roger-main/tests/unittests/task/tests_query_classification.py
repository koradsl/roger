from roger.task.query_classification import QueryClassificationTask
from tests.sample import openai_keys


def test_query_classification_task(openai_keys):
    openai_api_key, openai_org_key = openai_keys

    _, label = QueryClassificationTask(query="이 문서에 대해 설명해줘", api_key=openai_api_key, org_key=openai_org_key)
    assert isinstance(label, int)

    # TODO : label이 3이 아닌 경우가 있으니 주의해야 함
    assert label == 3
