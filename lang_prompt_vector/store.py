# Create a new weaviate vector database and store the result of the generated embeddings
import os
import weaviate
import streamlit as st

os.environ["OPENAI_API_KEY"] =""


@st.cache_data
def store_embeddings(embeddings):
    auth_config = weaviate.auth.AuthApiKey(api_key=st.secrets["WEAVIATE_API_KEY"])
    weaviate_client = weaviate.Client(
    url=st.secrets["WEAVIATE_CLUSTER_URL"],
    additional_headers={
        'X-OpenAI-Api-Key': os.environ["OPENAI_API_KEY"]
    },
    auth_client_secret=auth_config
    )
    
    
# def store_embeddings(embeddings):
#     client = weaviate.Client(
#         url=st.secrets["WEAVIATE_CLUSTER_URL"],
#         auth_client_secret=weaviate.AuthApiKey(
#             api_key=st.secrets["WEAVIATE_AUTH_KEY"]
#         ),
#     )

    try:
        weaviate_client.schema.delete_all()
        print("Schema deleted successfully...")
    except:
        print("Schema not deleted...")

    schema = {
        "classes": [
            {
                "class": "PDF",
                "moduleConfig": {
                    "text2vec-transformers": {
                        "skip": False,
                        "vectorizeClassName": False,
                        "vectorizePropertyName": False,
                    }
                },
                "vectorizer": "text2vec-openai",
                "properties": [
                    {
                        "name": "embeddings",
                        "dataType": ["number"],
                        "moduleConfig": {
                            "text2vec-transformers": {
                                "skip": False,
                                "vectorizePropertyName": False,
                                "vectorizeClassName": False,
                            }
                        },
                    }
                ],
            }
        ]
    }

    weaviate_client.schema.create(schema)
    print('Schema created...')

    weaviate_client.batch.configure(
        batch_size=10,
        dynamic=True,
        timeout_retries=3,
    )

    for i in range(0, len(embeddings)):
        item = embeddings.iloc[i]

        pdf_obj = {
            "embedded_values": item["embedded_values"],
        }

        try:
            weaviate_client.batch.add_data_object(pdf_obj, "PDF")
        except BaseException as error:
            print("Import Failed at: ", i)
            print("An exception occurred: {}".format(error))
            # Stop the import on error
            break

        print("Status: ", str(i) + "/" + str(len(embeddings) - 1))

    weaviate_client.batch.flush()
    print("Job done...")
    return True