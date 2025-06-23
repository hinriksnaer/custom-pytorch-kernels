from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="mlp_cuda",
    ext_modules=[
        CUDAExtension(
            name="mlp_cuda",
            sources=["extensions/cuda/mlp.cu", "extensions/cuda/mlp_binding.cpp"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
