from distutils.core import setup

setup(
    name='robot_ball_catcher',
    version='1.0.0',
    url='https://github.com/MatthiasDR96/robot_ball_catcher.git',
    license='BSD',
    author='Matthias De Ryck',
    author_email='matthias.deryck@kuleuven.be',
    description='software for the Robot Ball Catcher demo at the KU Leuven Campus in Bruges',
    packages=['robot_ball_catcher'],
    package_data={'robot_ball_catcher': ['data/*.npy']},
    package_dir={'': 'src'},
    scripts=['scripts/1_camera_viewer.py'],
    install_requires=['opencv-python', 'numpy', 'matplotlib']
)