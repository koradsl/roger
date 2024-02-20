from typing import Generator, List, Tuple

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

from roger.provider.oai.completion import OaiCompletion
from roger.provider.oai.embedding import OaiEmbedding


# Reference : https://pashpashpash.substack.com/p/tackling-the-challenge-of-document
class SummarizationTask:
    prompt: str = """Your task is to generate a brief summary of the document.
Summarize a given document, which must be up to 35 words."""

    @classmethod
    def __preprocess(
        cls,
        text: str,
        api_key: str,
        org_key: str,
        num_clusters: int = 8,
        max_iter: int = 3,
        chunk_size: int = 512,
        overlap_size: int = 100,
    ) -> str:
        """
        요약 전 전처리
        1. Text to chunks using chunk_size and overlap_size parameters
        2. Convert chunks to features
        3. Clustering using K-Means algorithm and select nearest text
        4. Merge selected texts
        """
        chunks = [text[idx : idx + chunk_size] for idx in range(0, len(text), chunk_size - overlap_size)]

        if len(chunks) == 1:
            return text

        feature_list = cls.__get_embeddings(chunks=chunks, api_key=api_key, org_key=org_key)

        closest_indexes = cls.__clustering(
            feature_list=feature_list,
            num_clusters=min(len(chunks), num_clusters),
            max_iter=max_iter,
        )

        closest_indexes.sort()

        return "\n".join([chunks[idx] for idx in closest_indexes])

    @classmethod
    def __get_embeddings(cls, chunks: List[str], api_key: str, org_key: str) -> List[List[float]]:
        """임베딩"""
        embedder = OaiEmbedding(api_key=api_key, org_key=org_key)

        _, response = embedder.call(inputs=chunks)

        return [data["features"] for data in response]

    @classmethod
    def __clustering(cls, feature_list: List[List[float]], num_clusters: int, max_iter: int):
        """Clustering(K-Means)"""
        cluster = KMeans(n_clusters=num_clusters, max_iter=max_iter)
        cluster.fit(feature_list)

        closest, _ = pairwise_distances_argmin_min(cluster.cluster_centers_, feature_list)

        return closest

    @classmethod
    def completion(cls, text: str, api_key: str, org_key: str) -> str:
        _, response = OaiCompletion(
            model_name="gpt-3.5-turbo-0613",
            api_key=api_key,
            org_key=org_key,
        ).call(
            messages=[{"role": "system", "content": cls.prompt}, {"role": "user", "content": text}],
            max_tokens=512,
        )

        return response

    @classmethod
    def stream(cls, text: str, api_key: str, org_key: str):
        # TODO : 구현 예정
        raise NotImplementedError()

    @classmethod
    def call(
        cls,
        text: str,
        api_key: str,
        org_key: str,
        # TODO : Token Size로 변경하는 것을 고려
        num_clusters: int = 8,
        max_iter: int = 3,
        chunk_size: int = 512,
        overlap_size: int = 100,
    ) -> Tuple[str, str, str]:
        text = cls.__preprocess(
            text=text,
            api_key=api_key,
            org_key=org_key,
            num_clusters=num_clusters,
            max_iter=max_iter,
            chunk_size=chunk_size,
            overlap_size=overlap_size,
        )
        return (
            cls.prompt,
            text,
            cls.completion(text=text, api_key=api_key, org_key=org_key),
        )

    @classmethod
    def stream(
        cls,
        text: str,
        api_key: str,
        org_key: str,
        num_clusters: int = 8,  # TODO : Token Size로 변경하는 것을 고려
        max_iter: int = 3,
        chunk_size: int = 512,
        overlap_size: int = 100,
    ) -> Tuple[str, str, Generator]:
        # text = cls.__preprocess(
        #     text=text,
        #     api_key=api_key,
        #     org_key=org_key,
        #     num_clusters=num_clusters,
        #     max_iter=max_iter,
        #     chunk_size=chunk_size,
        #     overlap_size=overlap_size,
        # )
        #
        # return (
        #     cls.profile,
        #     text,
        #     cls.__stream_summarization(text=text, api_key=api_key, org_key=org_key),
        # )
        # TODO : 구현 예정
        raise NotImplementedError()
