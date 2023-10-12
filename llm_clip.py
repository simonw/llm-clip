import llm
from PIL import Image
from sentence_transformers import SentenceTransformer
import io


@llm.hookimpl
def register_embedding_models(register):
    register(ClipEmbeddingModel())


class ClipEmbeddingModel(llm.EmbeddingModel):
    model_id = "clip"
    supports_binary = True
    supports_text = True

    def __init__(self):
        self._model = None

    def embed_batch(self, items):
        # Embeds a mix of text strings and binary images
        if self._model is None:
            self._model = SentenceTransformer("clip-ViT-B-32")

        to_embed = []

        for item in items:
            if isinstance(item, bytes):
                # If the item is a byte string, treat it as image data and convert to Image object
                to_embed.append(Image.open(io.BytesIO(item)))
            elif isinstance(item, str):
                to_embed.append(item)

        embeddings = self._model.encode(to_embed)
        return [[float(num) for num in embedding] for embedding in embeddings]
