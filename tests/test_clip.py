import llm


def test_run_embedding():
    model = llm.get_embedding_model("clip")
    result = model.embed("bunny")
    assert len(result) == 512
    assert isinstance(result[0], float)
