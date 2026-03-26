from setuptools import find_packages, setup

package_name = 'traj_interact_predict'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='wayne',
    maintainer_email='tslywh@nus.edu.sg',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'train = scripts.train:main',
            'inference = scripts.inference:main',
            'sim_agent = traj_interact_predict.simulation.sim_agent:main',
        ],
    },
)
