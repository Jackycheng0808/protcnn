import unittest
import torch
from backbone.layers.residual_se_block import ResidualSEBlock
from backbone.layers.residual_block import ResidualBlock
from backbone.resnet import ResNet, ResSENet
from backbone.minimscnn import MiniMSCNN
from backbone.mobilenet import MobileNetV2


class TestResidualSEBlock(unittest.TestCase):
    def test_type_shape(self):
        """check output type and shape"""
        inputs = torch.rand(22, 128, 128)
        model = ResidualSEBlock(128, 128, dilation=2)
        out = model(inputs)

        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, torch.Size([22, 128, 128]))

    def test_inputs(self):
        """check SEBlock working"""
        inputs = torch.zeros(22, 128, 128)
        model = ResidualSEBlock(128, 128, dilation=2)
        out = model(inputs)
        zero = torch.zeros(22, 128, 128)
        self.assertTrue(torch.equal(out, zero))


class TestResidualBlock(unittest.TestCase):
    def test_type_shape(self):
        """check output type and shape"""
        inputs = torch.rand(22, 128, 128)
        model = ResidualBlock(128, 128, dilation=2)
        out = model(inputs)

        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, torch.Size([22, 128, 128]))

    def test_inputs(self):
        """check ResnetBlock working"""
        inputs = torch.zeros(22, 128, 128)
        model = ResidualBlock(128, 128, dilation=2)
        out = model(inputs)
        zero = torch.zeros(22, 128, 128)
        self.assertTrue(torch.equal(out, zero))


class TestResidualNet(unittest.TestCase):
    def test_type_shape(self):
        """check output type and shape"""
        inputs = torch.rand(1, 22, 120)
        model = ResNet()
        out = model(inputs)

        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, torch.Size([1, 17930]))


class TestResidualSENet(unittest.TestCase):
    def test_type_shape(self):
        """check output type and shape"""
        inputs = torch.rand(1, 22, 120)
        model = ResSENet()
        out = model(inputs)

        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, torch.Size([1, 17930]))


class TestMobileNet(unittest.TestCase):
    def test_type_shape(self):
        """check output type and shape"""
        inputs = torch.rand(1, 22, 120)
        model = MobileNetV2()
        out = model(inputs)

        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, torch.Size([1, 17930]))


if __name__ == "__main__":
    unittest.main()
