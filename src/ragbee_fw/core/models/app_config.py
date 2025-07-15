from typing import Annotated, Any

from pydantic import BaseModel, Field


class DataLoader(BaseModel):
    type: Annotated[str, "Type of data Loader adapters", "Implimenting DataLoaderPort"]
    path: Annotated[str, "Path for loaded documents"]
    params: dict[str, Any] | None = Field(
        default=None,
        description="""
                        A dictionary of parameters not provided by the implemented 
                        adapters in the framework. 
                        Necessary for custom adapter implementations 
                        and future scaling of the framework.
                        """,
    )


class Splitter(BaseModel):
    type: Annotated[
        str,
        "A type of spitter that breaks a document into chunks",
        "Implimenting TextSplitterPort",
    ]
    chunk_size: int = Field(
        default=1000,
        description="""
                            The approximate size of one piece of the source 
                            document that will be searched
                            """,
    )
    chunk_overlap: int = Field(
        default=200,
        description="""
                            The overlap size of adjacent chunks in characters
                            """,
    )
    separators: list[str] | None = Field(
        default_factory=list,
        description="""
                        Separators that most likely divide sections of text 
                        into logically complete sections that can be allocated into chunks.
                        """,
    )
    params: dict[str, Any] | None = Field(
        default=None,
        description="""
                        A dictionary of parameters not provided by the implemented 
                        adapters in the framework. 
                        Necessary for custom adapter implementations 
                        and future scaling of the framework.
                        """,
    )


class Retriever(BaseModel):
    type: Annotated[
        str,
        """A type of retriever that will find the most similar chunks 
            either by full-text search, or by semantic similarity, or a hybrid""",
        "Implimenting RetrieverPort",
    ]
    top_k: int = Field(
        default=5,
        description="""
                        The number of text chunks output/transmitted to LLM 
                        that are most relevant for supplementing the information 
                        in response to the user's original request.
                        """,
    )
    params: dict[str, Any] | None = Field(
        default=None,
        description="""
                        A dictionary of parameters not provided by the implemented 
                        adapters in the framework. 
                        Necessary for custom adapter implementations 
                        and future scaling of the framework.
                        """,
    )


class LLM(BaseModel):
    type: Annotated[
        str,
        """
            Type of LLM provider class implemented as an adapter
            """,
        "Implimenting LLMPort",
    ]
    model_name: Annotated[
        str,
        """
            Model name/repository+model name (if necessary)
            """,
    ]
    token: str | None = Field(
        default=None,
        description="""
                        Access token to the model provider if needed. 
                        In case of using the local LLM model, the token can be set as None.
                        """,
    )
    provider: str | None = Field(
        default="auto",
        description="""
                        The direct name of the provider providing access to the LLM model 
                        within the provider class.
                        """,
    )
    base_url: str | None = Field(
        default=None,
        description="""
                        URL for accessing the LLM API. 
                        Most often needed for direct access to the chat complete method. 
                        But it can also be used as a basic component for various methods, 
                        including checking the health of the hosting server and the model itself, 
                        as well as receiving method data, transmitting inference server settings, etc.
                        """,
    )
    prompt: str = Field(
        default="""You are an intelligent assistant. 
                    Answer the question based on additional information found 
                    on the topic of the question in the user knowledge base.
                    """,
        description="""
                        System prompt for the original LLM task - answer to the user's question 
                        based on the relevant document chunks found.
                        """,
    )
    max_new_tokens: int = Field(
        default=2048,
        description="""
                        The number of model output tokens 
                        calculated for the most verbose response from the LLM.
                        """,
    )
    return_full_response: bool = Field(
        default=False,
        description="""
                        Parameter that determines whether to return the text/content of the response
                        from LLM or a tuple of tuple(text, full response with metadata)
                        """,
    )
    params: dict[str, Any] | None = Field(
        default=None,
        description="""
                        A dictionary of parameters not provided by the implemented 
                        adapters in the framework. 
                        Necessary for custom adapter implementations 
                        and future scaling of the framework.
                        """,
    )


class AppConfig(BaseModel):
    data_loader: DataLoader
    text_chunker: Splitter
    retriever: Retriever
    llm: LLM
