from src.indexing import tokenize


def test_tokenize_keeps_hindi_words_and_drops_function_words():
    tokens = tokenize("कॉर्पोरेशन क्या है?")
    assert "कॉर्पोरेशन" in tokens
    assert "क्या" not in tokens
    assert "है" not in tokens
