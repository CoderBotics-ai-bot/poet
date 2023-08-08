from build.lib.poet.util import *
def test_get_chipset_and_net():
    # Initialize instances from classes available in scope and parameters values
    test_instances = [
        ("m0", "linear", 1, MKR1000, type(make_linear_network())),
        ("a72", "vgg16", 2, RPi, type(vgg16(2))),
        ("a72nocache", "resnet18", 3, RPiNoCache, type(resnet18(3))),
        ("m4", "resnet50", 4, M4F, type(resnet50(4))),
        (
            "jetsontx2",
            "bert",
            5,
            JetsonTX2,
            type(
                BERTBase(
                    SEQ_LEN=512,
                    HIDDEN_DIM=768,
                    I=64,
                    HEADS=12,
                    NUM_TRANSFORMER_BLOCKS=12,
                )
            ),
        ),
    ]

    for (
        platform,
        model,
        batch_size,
        expected_chipset,
        expected_net_class,
    ) in test_instances:
        chipset, net = get_chipset_and_net(platform, model, batch_size)
        assert isinstance(chipset, type(expected_chipset))
        assert isinstance(net, expected_net_class)

    # Test invalid platform
    with pytest.raises(NotImplementedError):
        get_chipset_and_net("invalid_platform", "resnet50", 4)

    # Test invalid model
    with pytest.raises(NotImplementedError):
        get_chipset_and_net("m4", "invalid_model", 4)

    # Assert that the MEMORY_POWER value of the chipset is modified when mem_power_scale is given
    initial_memory_power = MKR1000["MEMORY_POWER"]
    chipset, _ = get_chipset_and_net("m0", "linear", 1, mem_power_scale=1.5)
    assert chipset["MEMORY_POWER"] == initial_memory_power * 1.5


from poet.architectures.linear import make_linear_network
from poet.architectures.vgg import vgg16
from poet.architectures.resnet import resnet18, resnet50
from poet.architectures.bert import BERTBase
from poet.chipsets import M4F, MKR1000, JetsonTX2, RPi, RPiNoCache
import pytest
