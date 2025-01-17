from typing import Optional, List, Dict, Any
from griptape.drivers import BaseVectorStoreDriver
from griptape.artifacts import TextArtifact
import marqo
from attr import define, field, Factory
import logging


@define
class MarqoVectorStoreDriver(BaseVectorStoreDriver):
    api_key: str = field(kw_only=True)
    url: str = field(kw_only=True)
    mq: marqo.Client = field(
        default=Factory(
            lambda self: marqo.Client(self.url, self.api_key), takes_self=True
        ),
        kw_only=True,
    )
    index: str = field(kw_only=True)

    def __attrs_post_init__(self):
        """Initialize the Marqo client with the given API key and URL."""
        self.set_index(self.index)

    def set_index(self, index):
        """Set the index for the Marqo client.

        Args:
            index (str): The index to set for the Marqo client.
        """
        indexes = self.get_indexes()

        if index not in indexes:
            self.create_index(index)
            logging.info(f"Created index '{index}'")

        self.index = index

    def upsert_text(
            self,
            string: str,
            vector_id: Optional[str] = None,
            namespace: Optional[str] = None,
            meta: Optional[dict] = None,
            **kwargs
    ) -> str:
        """Upsert a text document into the Marqo index.

        Args:
            string (str): The string to be indexed.
            vector_id (Optional[str], optional): The ID for the vector. If None, Marqo will generate an ID.
            namespace (Optional[str], optional): An optional namespace for the document.
            meta (Optional[dict], optional): An optional dictionary of metadata for the document.

        Returns:
            str: The ID of the document that was added.
        """

        doc = {
            "_id": vector_id,
            "Description": string,  # Description will be treated as tensor field
        }

        # Non-tensor fields
        if meta:
            doc["meta"] = str(meta)
        if namespace:
            doc['namespace'] = namespace

        response = self.mq.index(self.index).add_documents([doc], non_tensor_fields=["meta", "namespace"])
        return response["items"][0]["_id"]

    def upsert_text_artifact(
            self,
            artifact: TextArtifact,
            namespace: Optional[str] = None,
            meta: Optional[dict] = None,
            **kwargs
    ) -> str:
        """Upsert a text artifact into the Marqo index.

        Args:
            artifact (TextArtifact): The text artifact to be indexed.
            namespace (Optional[str], optional): An optional namespace for the artifact.
            meta (Optional[dict], optional): An optional dictionary of metadata for the artifact.

        Returns:
            str: The ID of the artifact that was added.
        """

        artifact_json = artifact.to_json()

        doc = {
            "_id": artifact.id,
            "Description": artifact.value,  # Description will be treated as tensor field
            "artifact": str(artifact_json),
            "namespace": namespace
        }

        response = self.mq.index(self.index).add_documents([doc], non_tensor_fields=["meta", "namespace", "artifact"])
        return response["items"][0]["_id"]

    def load_entry(self, vector_id: str, namespace: Optional[str] = None) -> Optional[BaseVectorStoreDriver.Entry]:
        """Load a document entry from the Marqo index.

        Args:
            vector_id (str): The ID of the vector to load.
            namespace (Optional[str], optional): The namespace of the vector to load.

        Returns:
            Optional[BaseVectorStoreDriver.Entry]: The loaded Entry if found, otherwise None.
        """
        result = self.mq.index(self.index).get_document(document_id=vector_id, expose_facets=True)

        if result and "_tensor_facets" in result and len(result["_tensor_facets"]) > 0:
            return BaseVectorStoreDriver.Entry(
                id=result["_id"],
                meta={k: v for k, v in result.items() if k not in ["_id"]},
                vector=result["_tensor_facets"][0]["_embedding"],
            )
        else:
            return None

    def load_entries(self, namespace: Optional[str] = None) -> list[BaseVectorStoreDriver.Entry]:
        """Load all document entries from the Marqo index.

        Args:
            namespace (Optional[str], optional): The namespace to filter entries by.

        Returns:
            list[BaseVectorStoreDriver.Entry]: The list of loaded Entries.
        """

        filter_string = f"namespace:{namespace}" if namespace else None
        results = self.mq.index(self.index).search("", limit=10000, filter_string=filter_string)

        # get all _id's from search results
        ids = [r["_id"] for r in results["hits"]]

        # get documents corresponding to the ids
        documents = self.mq.index(self.index).get_documents(document_ids=ids, expose_facets=True)

        # for each document, if it's found, create an Entry object
        entries = []
        for doc in documents['results']:
            if doc['_found']:
                entries.append(
                    BaseVectorStoreDriver.Entry(
                        id=doc["_id"],
                        vector=doc["_tensor_facets"][0]["_embedding"],
                        meta={k: v for k, v in doc.items() if k not in ["_id", "_tensor_facets", "_found"]},
                        namespace=doc.get("namespace"),
                    )
                )

        return entries

    def query(
            self,
            query: str,
            count: Optional[int] = None,
            namespace: Optional[str] = None,
            include_vectors: bool = False,
            include_metadata=True,
            **kwargs
    ) -> list[BaseVectorStoreDriver.QueryResult]:
        """Query the Marqo index for documents.

        Args:
            query (str): The query string.
            count (Optional[int], optional): The maximum number of results to return.
            namespace (Optional[str], optional): The namespace to filter results by.
            include_vectors (bool, optional): Whether to include vector data in the results.
            include_metadata (bool, optional): Whether to include metadata in the results.

        Returns:
            list[BaseVectorStoreDriver.QueryResult]: The list of query results.
        """

        params = {
                     "limit": count if count else BaseVectorStoreDriver.DEFAULT_QUERY_COUNT,
                     "attributes_to_retrieve": ["*"] if include_metadata else ["_id"],
                     "filter_string": f"namespace:{namespace}" if namespace else None
                 } | kwargs

        results = self.mq.index(self.index).search(query, **params)

        if include_vectors:
            results["hits"] = list(map(lambda r: self.mq.index(self.index).get_document(r["_id"]), results["hits"]))

        return [
            BaseVectorStoreDriver.QueryResult(
                vector=[],
                score=r["_score"],
                meta={k: v for k, v in r.items() if k not in ["_score"]},
            )
            for r in results["hits"]
        ]

    def create_index(self, name: str, **kwargs) -> Dict[str, Any]:
        """Create a new index in the Marqo client.

        Args:
            name (str): The name of the new index.
        """

        return self.mq.create_index(name, settings_dict=kwargs)

    def delete_index(self, name: str) -> Dict[str, Any]:
        """Delete an index in the Marqo client.

        Args:
            name (str): The name of the index to delete.
        """

        return self.mq.delete_index(name)

    def get_indexes(self) -> List[str]:
        """Get a list of all indexes in the Marqo client.

        Returns:
            list: The list of all indexes.
        """

        # Change this once API issue is fixed (entries in results are no longer objects but dicts)
        return [index.index_name for index in self.mq.get_indexes()["results"]]

    def upsert_vector(
            self,
            vector: list[float],
            vector_id: Optional[str] = None,
            namespace: Optional[str] = None,
            meta: Optional[dict] = None,
            **kwargs
    ) -> str:
        """Upsert a vector into the Marqo index.

        Args:
            vector (list[float]): The vector to be indexed.
            vector_id (Optional[str], optional): The ID for the vector. If None, Marqo will generate an ID.
            namespace (Optional[str], optional): An optional namespace for the vector.
            meta (Optional[dict], optional): An optional dictionary of metadata for the vector.

        Raises:
            Exception: This function is not yet implemented.

        Returns:
            str: The ID of the vector that was added.
        """

        raise Exception("not implemented")
