## Install packages
You should install the packages to use PDAL:
```bash
sudo apt-get install libpdal-dev pdal
```

## Running the code
To compile and run the code, follow these steps:
1. Navigate to the `cudaproject` directory:
   ```bash
   cd project/cudaproject
   ```
2. Create a build directory and navigate into it:
   ```bash
   mkdir build && cd build
   ```
3. Run CMake to configure the project:
   ```bash
   cmake ..
   ```
4. Build the project using Make:
   ```bash
   make
   ```
5. Run the executable with a LAS file as an argument:
   ```bash
   ./cudaproject path_lasfile.las
   ```

## Documentations
- [PDAL documentation](https://pdal.io/)