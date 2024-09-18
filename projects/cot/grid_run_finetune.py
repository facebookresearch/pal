"""
Example of fine-tuning grid run.

License
-------
This source code is licensed under the CC license found in the LICENSE file
in the root directory of this source tree.

@ 2024, Meta
"""

import json
import logging
import traceback
from dataclasses import asdict, dataclass
from itertools import product
from uuid import uuid4

from cot.config import CHECK_DIR, DATA_DIR
from cot.data import data_processing
from cot.transfer import transfer
from cot.utils import JsonEncoder

logger = logging.getLogger(__name__)


@dataclass
class MainConfig:
    # Problem
    problem: str = "binary-copy"

    # Checkpointed model and fine-tuned layers
    load_path: str = None
    finetune_mlp2: bool = False

    # Data
    data_dir: str = None
    n_len: int = 16
    split_probas: float = 0.5
    n_data_per_len: int = 1024

    # Optimization
    n_epochs: int = 1000
    batch_size: int = 256
    learning_rate: float = 3e-4

    # Extra optimization option
    emb_dropout: float = 0.1

    # Checkpointing
    checkpoint: bool = False
    checkpoint_freq: int = 100
    overwrite_checkpoint: bool = True
    check_dir: str = None

    # Evaluation
    full_eval: bool = True
    eval_freq: int = 10

    # Run id
    run_id: int = 0

    def __post_init__(self):
        self.unique_id = str(uuid4())
        if self.data_dir == "special":
            self.data_dir = DATA_DIR / self.unique_id

        if self.check_dir == "special":
            self.check_dir = CHECK_DIR / self.unique_id


def run_experiment(
    config,
    run_data=True,
    run_train=True,
):
    """
    Run one experiments associated with one config file

    Parameters
    ----------
    config: Config class
    """

    if run_data:
        data_processing(
            problem=config.problem,
            n_len=config.n_len,
            split_probas=config.split_probas,
            n_data_per_len=config.n_data_per_len,
            save_dir=config.data_dir,
        )

    if run_train:
        transfer(
            load_path=config.load_path,
            problem=config.problem,
            finetune_mlp2=config.finetune_mlp2,
            data_dir=config.data_dir,
            n_len=config.n_len,
            n_epochs=config.n_epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            emb_dropout=config.emb_dropout,
            checkpoint=config.checkpoint,
            checkpoint_freq=config.checkpoint_freq,
            overwrite_checkpoint=config.overwrite_checkpoint,
            check_dir=config.check_dir,
            full_eval=config.full_eval,
            eval_freq=config.eval_freq,
        )


def run_grid(
    num_tasks=1,
    task_id=1,
    config_filename=None,
):
    """
    Run a grid of experiments

    Parameters
    ----------
    num_tasks: int
        Number of threads to split the grid run into.
    task_id: int
        Id of the current thread.
    config_filename: str
        Where to save the configuration that generate the runs.
    """

    grid = {
        "problem": ["parity"],
        "load_path": [
            "/checkpoint/vivc/models/62a4398a-0b5e-4d12-84c0-840e4146d5ac/model_200.pth",
            "/checkpoint/vivc/models/971a342d-68d2-40e2-9d27-72681caab92e/model_200.pth",
            "/checkpoint/vivc/models/92a2897f-a072-4f75-941d-4a93bb919600/model_200.pth",
            "/checkpoint/vivc/models/4f651b9f-ae2a-45df-b6da-d100eb3cebea/model_200.pth",
            "/checkpoint/vivc/models/aae31bdc-88f2-49be-a9e8-b93b3f729153/model_200.pth",
            "/checkpoint/vivc/models/d97ad3f5-b585-4bfe-920d-4b31ffbacbba/model_200.pth",
            "/checkpoint/vivc/models/81de937d-a78c-4088-90ef-1d5869ab1736/model_200.pth",
            "/checkpoint/vivc/models/759b2282-e5cc-4ffb-a468-05748f1465be/model_200.pth",
            "/checkpoint/vivc/models/cf5ae8ff-7ade-4017-ae98-994c23fcada2/model_200.pth",
            "/checkpoint/vivc/models/d9dea382-65be-405b-b756-ec765804d481/model_200.pth",
            "/checkpoint/vivc/models/7a3f3fb0-4b55-4678-bf19-a2f0f65b479e/model_200.pth",
            "/checkpoint/vivc/models/cd9e1bbe-c103-451e-a249-87732a052563/model_200.pth",
            "/checkpoint/vivc/models/d55d6d26-fe57-4b8e-8657-c04ef8031f3c/model_200.pth",
            "/checkpoint/vivc/models/0d5929d3-d6b3-4089-93a8-9c9859f879b8/model_200.pth",
            "/checkpoint/vivc/models/ed54f9e1-0c99-4511-82ff-82c22b12cdf2/model_200.pth",
            "/checkpoint/vivc/models/9ccb42ec-a5d6-40bb-9803-537baaaf4c63/model_200.pth",
            "/checkpoint/vivc/models/dee8926f-692b-4c07-a5d2-f83561c1d509/model_200.pth",
            "/checkpoint/vivc/models/cee63a6b-04ee-48a5-b4de-b40eee996c4f/model_200.pth",
            "/checkpoint/vivc/models/d55eb26d-32db-4ad4-af9c-7813b02913e2/model_200.pth",
            "/checkpoint/vivc/models/6feec2c0-30e0-4caa-96c3-0dad43060dff/model_200.pth",
            "/checkpoint/vivc/models/4b50f8d4-79da-4838-a8c1-70b7c15e8efb/model_200.pth",
            "/checkpoint/vivc/models/9ba7be3d-ec3a-4eae-a00f-746fbe4b0e04/model_200.pth",
            "/checkpoint/vivc/models/492e04f8-7ddd-49db-8db5-112dafcd77d5/model_200.pth",
            "/checkpoint/vivc/models/866d2d4c-73f6-4807-b454-260c9fdce027/model_200.pth",
            "/checkpoint/vivc/models/c314b324-d70f-443d-bfcb-3cd4f5cc0880/model_200.pth",
            "/checkpoint/vivc/models/85324f9c-0e38-480d-8df7-aa1561dfc7cd/model_200.pth",
            "/checkpoint/vivc/models/0154c997-23f8-43df-b667-e8ae877ecae8/model_200.pth",
            "/checkpoint/vivc/models/7945a23b-87f2-4320-9559-f5a5123056cd/model_200.pth",
            "/checkpoint/vivc/models/ff729bc3-7200-4fc4-b97c-6b3faaee850f/model_200.pth",
            "/checkpoint/vivc/models/e42544d1-ef98-4be4-a464-8a2f4802ac2d/model_200.pth",
            "/checkpoint/vivc/models/304fc2a3-4255-4bf0-9696-9fc5d697fe0e/model_200.pth",
            "/checkpoint/vivc/models/70048e5d-7f4e-4d28-aa73-f8e4ecc2b179/model_200.pth",
            "/checkpoint/vivc/models/061eebc6-8027-4f1d-a306-947e88e25f8a/model_200.pth",
            "/checkpoint/vivc/models/e8dc3203-0111-4ef8-a899-d9fd04d18c52/model_200.pth",
            "/checkpoint/vivc/models/438c967f-bfc8-4050-ab7b-1869e41a5266/model_200.pth",
            "/checkpoint/vivc/models/c29d62e3-9deb-4e7f-b7f3-e5b2cd7ccafb/model_200.pth",
            "/checkpoint/vivc/models/a67e2546-8ec9-4527-ad11-cfa33405ed24/model_200.pth",
            "/checkpoint/vivc/models/3fa2821c-1d5f-4659-a5b6-18333ff92b31/model_200.pth",
            "/checkpoint/vivc/models/f20228a6-cc40-4af4-b790-9da0bd4b4cdf/model_200.pth",
            "/checkpoint/vivc/models/33a9a23a-6bc6-4776-8f73-cee87e72eda5/model_200.pth",
            "/checkpoint/vivc/models/f0b3d42b-9e8c-4eee-997c-a0aeadd9db6b/model_200.pth",
            "/checkpoint/vivc/models/18f6f5d0-d209-401a-9e76-ebd899028d6e/model_200.pth",
            "/checkpoint/vivc/models/934e9a1c-82f7-4f3c-9265-757e920e825e/model_200.pth",
            "/checkpoint/vivc/models/a0ff7a1e-ecf1-4012-9a9f-d82b49925194/model_200.pth",
            "/checkpoint/vivc/models/289d23c7-80fd-454d-a5b7-ed09b50b438e/model_200.pth",
            "/checkpoint/vivc/models/cc4dbbb0-c8a3-4a9b-a967-9c890a13f22f/model_200.pth",
            "/checkpoint/vivc/models/173c65fe-d536-4d14-913c-93b6576723da/model_200.pth",
            "/checkpoint/vivc/models/c33ae586-e29d-490b-94bd-d92211e11698/model_200.pth",
            "/checkpoint/vivc/models/d3f5e532-1332-45c4-af55-4dd6015a2397/model_200.pth",
            "/checkpoint/vivc/models/bb64f378-7383-4b9d-96f2-4cfb64253286/model_200.pth",
            "/checkpoint/vivc/models/663a5948-e825-4890-b0f8-366990e1ae00/model_200.pth",
            "/checkpoint/vivc/models/03a6ef4c-eee2-4a76-b448-494f207a9a7b/model_200.pth",
            "/checkpoint/vivc/models/c637219f-3984-45e1-b91d-0505f97f508e/model_200.pth",
            "/checkpoint/vivc/models/4f314215-179f-4f50-8429-6f6c49b67efd/model_200.pth",
            "/checkpoint/vivc/models/e90da978-80ea-401e-8718-fee7bad58466/model_200.pth",
            "/checkpoint/vivc/models/e6e86bf4-3e21-4d12-8317-1b9c83b760ef/model_200.pth",
            "/checkpoint/vivc/models/fc131433-f851-4faf-931f-af2f9aca5d05/model_200.pth",
            "/checkpoint/vivc/models/77fc5079-8bab-4da7-9294-48fdba54c786/model_200.pth",
            "/checkpoint/vivc/models/977d7fb2-e681-493e-822e-23e2b502f50e/model_200.pth",
            "/checkpoint/vivc/models/b3438f44-faea-4aa1-ad87-ca461772da97/model_200.pth",
            "/checkpoint/vivc/models/97cac078-489e-4115-b6a5-c798f84f937b/model_200.pth",
            "/checkpoint/vivc/models/4fd4bd53-4d8f-4121-a2a8-8a0b76ae09b6/model_200.pth",
            "/checkpoint/vivc/models/b0f53894-6b9c-4c38-bfa3-ae48300a415b/model_200.pth",
            "/checkpoint/vivc/models/6c030bf5-afdd-4cb1-b051-e42eaab9ebb9/model_200.pth",
            "/checkpoint/vivc/models/73ec0f9f-5fe2-411a-b265-c9c54b28a602/model_200.pth",
            "/checkpoint/vivc/models/20177112-1f9f-4df2-8d24-7cb2f6bda0ac/model_200.pth",
            "/checkpoint/vivc/models/43dda6ed-913a-4c82-93d8-5b52c44c2f52/model_200.pth",
            "/checkpoint/vivc/models/ff494218-3043-4385-9a77-6b09bdfb1165/model_200.pth",
            "/checkpoint/vivc/models/552870ea-47f9-4ae5-8720-6d82dd9100ed/model_200.pth",
            "/checkpoint/vivc/models/9637ec27-8d4e-4341-9839-4041bbd2e3e0/model_200.pth",
            "/checkpoint/vivc/models/03f2a64d-f5d5-4613-9033-630fa169b9a5/model_200.pth",
            "/checkpoint/vivc/models/a0a0f70f-4a17-49d5-a828-28581ef95131/model_200.pth",
            "/checkpoint/vivc/models/16a6b037-1c02-4622-8818-f11f16cf3c4c/model_200.pth",
            "/checkpoint/vivc/models/35f36700-7d31-46d3-a478-3a4eefa9855f/model_200.pth",
            "/checkpoint/vivc/models/838157db-9ed6-43f8-864e-688718e3365e/model_200.pth",
            "/checkpoint/vivc/models/ca890b45-f21d-4ff8-b298-af8a1581c466/model_200.pth",
            "/checkpoint/vivc/models/65cb5cee-2035-485a-8bb4-96d98499cb3c/model_200.pth",
            "/checkpoint/vivc/models/bac7ecc5-d233-40f3-a02b-5e78310a2348/model_200.pth",
            "/checkpoint/vivc/models/c7f25dc5-4651-4840-ac76-17372eac83e9/model_200.pth",
            "/checkpoint/vivc/models/a3f7a87e-2438-42c3-966e-0398885ba120/model_200.pth",
            "/checkpoint/vivc/models/9140d38f-365c-4c31-8aa5-5d69395efb6f/model_200.pth",
            "/checkpoint/vivc/models/50afcbaf-29be-4644-b64d-90f8ccb19c46/model_200.pth",
            "/checkpoint/vivc/models/e75c44a4-4231-4c81-b8c8-a7935258c32f/model_200.pth",
            "/checkpoint/vivc/models/f6a8d09f-9ceb-4a3e-aee0-8663ebded0cb/model_200.pth",
            "/checkpoint/vivc/models/4963d88b-1d98-4a4f-b4d6-dbaba7d96b80/model_200.pth",
            "/checkpoint/vivc/models/f3eadf41-a1e8-4987-936a-25caa4b9af3f/model_200.pth",
            "/checkpoint/vivc/models/81cff3b9-0976-40e6-9a9f-ce7d4880a875/model_200.pth",
            "/checkpoint/vivc/models/f82fb62f-2968-4b87-a66f-826ccd48adf7/model_200.pth",
            "/checkpoint/vivc/models/4f30ec59-557e-483d-8518-21e1ff830a4a/model_200.pth",
            "/checkpoint/vivc/models/bdf2c5aa-a699-439a-936f-f356f86c3d94/model_200.pth",
            "/checkpoint/vivc/models/36794a60-e2c3-4435-ba2f-65ed1702d239/model_200.pth",
            "/checkpoint/vivc/models/228804b7-e189-46e3-aa8b-f9675fefbe81/model_200.pth",
            "/checkpoint/vivc/models/26bb3856-4338-4086-87ca-883a47122818/model_200.pth",
            "/checkpoint/vivc/models/7d7b85a3-eddb-48d8-ba4b-f91ac44d1b07/model_200.pth",
            "/checkpoint/vivc/models/51c2d7d9-62ae-420c-92ce-a135f71a34c2/model_200.pth",
            "/checkpoint/vivc/models/ee98f41f-8741-46a1-9955-c466e711360c/model_200.pth",
            "/checkpoint/vivc/models/f56bb828-7d78-4e71-8c0e-dfd0606c70d4/model_200.pth",
            "/checkpoint/vivc/models/1211ce65-298b-42b0-a096-78c317846f14/model_200.pth",
        ],
    }

    CHECK_DIR.mkdir(parents=True, exist_ok=True)

    if config_filename is None:
        config_filename = "config"

    for i, values in enumerate(product(*grid.values())):
        # Handling the grid concurrently with many tasks
        if i % num_tasks != (task_id - 1):
            continue

        config = MainConfig(
            check_dir="special",
            data_dir="special",
            run_id=i,
        )

        for k, v in zip(grid.keys(), values):
            setattr(config, k, v)

        config_dict = asdict(config)
        with open(CHECK_DIR / f"{config_filename}.jsonl", "a") as f:
            json.dump(config_dict, f, cls=JsonEncoder, indent=4)
            f.write("\n")

        logger.info(f"{config=}")

        try:
            run_experiment(config)
        except Exception as e:
            logger.warning(f"Error for configuration: {config}.")
            logger.warning(traceback.format_exc())
            logger.warning(e)
            continue


if __name__ == "__main__":
    import fire

    from cot.config import logging_datefmt, logging_format, logging_level

    logging.basicConfig(
        format=logging_format,
        datefmt=logging_datefmt,
        style="{",
        level=logging_level,
        handlers=[logging.StreamHandler()],
    )

    fire.Fire(run_grid)
