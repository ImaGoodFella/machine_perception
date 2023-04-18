from smplx import MANO

MANO_MODEL_DIR = "./data/mano"


def build_mano(is_rhand):
    return MANO(
        MANO_MODEL_DIR,
        create_transl=False,
        use_pca=False,
        flat_hand_mean=False,
        is_rhand=is_rhand,
    )
