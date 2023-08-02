from src.utils import load_model

NORMAL_LIST= [239,486,0,8,8,19,0,19]
DOS_LIST= [0,0,0,271,17,255,0,17]
PROBE_LIST= [0,0,0,38,1,208,0,1]
R2L_LIST= [1237,2451,1,1,1,228,28,56]
U2R_LIST= [1511,2957,184,1,1,1,3,3]

def test_pipeline_label(pipeline, LIST, target):
    model = load_model(pipeline)
    df=[LIST]
    prediction = model.predict(df)
    if prediction[0] == target:
        print("Test Passed for" + target)
        return prediction
    else:
        raise Exception("Test Failed for" + target)

def test_pipeline(pipeline):
    """Tests pipeline

    Keyword arguments:
    pipeline -- loaded from outputs/pipeline.pkl

    Return: Success if tests passess
    """
    test_pipeline_label(pipeline, NORMAL_LIST, "Normal")
    test_pipeline_label(pipeline, DOS_LIST, "DoS")
    test_pipeline_label(pipeline, PROBE_LIST, "Probe")
    test_pipeline_label(pipeline, R2L_LIST, "R2L")
    test_pipeline_label(pipeline, U2R_LIST, "U2R")
    return 0
