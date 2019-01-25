import setuptools

with open('README.md', 'r') as f:
  long_description = f.read()
	
with open('LICENSE', 'r') as f:
    license = f.read()
  
setuptools.setup(
    name="basketball_stats-lbianculli",
    version="0.1.0",
    author="Luke Bianculli",
    author_email="lbianculli123@gmail.com",
    description="A small package for collecting and visualizing public basketball data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lbianculli/basketball_stats
    packages=setuptools.find_packages(exclude=('docs')),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
