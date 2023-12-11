#!/bin/bash

plat_headfile="./include/plat_config.h"

# Create the directory if it doesn't exist
# mkdir -p $(dirname "$plat_headfile")

# Function to convert cache size to KiB
convert_to_bytes() {
    size=$1
    unit=$2
    # Convert to Bytes based on the unit
    case $unit in
        K|kB|KiB)
            echo $(($size * 1024))
            ;;
        M|MB|MiB)
            echo $(($size * 1024 * 1024))
            ;;
        G|GB|GiB)
            echo $(($size * 1024 * 1024 * 1024))
            ;;
        *)
            echo $size  # Assume Bytes if no unit
            ;;
    esac
}

# Get CPU details
cpu_socket=$(lscpu | grep "Socket(s):" | awk '{print $2}')
cpu_cores=$(lscpu | grep "Core(s) per socket:" | awk '{print $4}')

cpu_freq_raw=$(lscpu | grep "CPU MHz:" | awk '{print $3}')

cpu_max_freq_raw=$(lscpu | grep "CPU max MHz:" | awk '{print $4}')
cpu_hyper_thread=$(lscpu | grep "Thread(s) per core:" | awk '{print $4}')
numa_region=$(lscpu| grep "NUMA node(s):" | awk '{print $3}')

# Extract L3 cache size and unit
cpu_l3cache_raw=$(lscpu | grep "L3 cache:" | awk '{print $3}' | sed 's/[^0-9]*//g')
cpu_l3cache_unit=$(lscpu | grep "L3 cache:" | awk '{print $4}')
cpu_l3cache_bytes=$(convert_to_bytes $cpu_l3cache_raw $cpu_l3cache_unit)

# Extract L2 cache size and unit
cpu_l2cache_raw=$(lscpu | grep "L2 cache:" | awk '{print $3}' | sed 's/[^0-9]*//g')
cpu_l2cache_unit=$(lscpu | grep "L2 cache:" | awk '{print $4}')
cpu_l2cache_bytes=$(convert_to_bytes $cpu_l2cache_raw $cpu_l2cache_unit)

# Extract L1 data cache size and unit
cpu_l1d_cache_raw=$(lscpu | grep "L1d cache:" | awk '{print $3}' | sed 's/[^0-9]*//g')
cpu_l1d_cache_unit=$(lscpu | grep "L1d cache:" | awk '{print $4}')
cpu_l1d_cache_bytes=$(convert_to_bytes $cpu_l1d_cache_raw $cpu_l1d_cache_unit)

# Extract L1 instruction cache size and unit
cpu_l1i_cache_raw=$(lscpu | grep "L1i cache:" | awk '{print $3}' | sed 's/[^0-9]*//g')
cpu_l1i_cache_unit=$(lscpu | grep "L1i cache:" | awk '{print $4}')
cpu_l1i_cache_bytes=$(convert_to_bytes $cpu_l1i_cache_raw $cpu_l1i_cache_unit)

# Extract Main Memory size
total_mem=$(grep MemTotal /proc/meminfo | awk '{print $2}') # KB
total_mem_GB=$(($total_mem / 1024 / 1024))

# Prepare the header file content
echo "#ifndef CONFIG_H" > $plat_headfile
echo "#define CONFIG_H" >> $plat_headfile
echo "" >> $plat_headfile
echo "#define CPU_FREQUENCY ${cpu_freq_raw}e6" >> $plat_headfile
echo "#define CPU_MAX_FREQUENCY ${cpu_max_freq_raw}e6" >> $plat_headfile
echo "#define CPU_SOCKET $cpu_socket" >> $plat_headfile
echo "#define CPU_CORES_PER_SOC $cpu_cores" >> $plat_headfile
echo "#define CPU_HYPER_THREAD $cpu_hyper_thread" >> $plat_headfile
echo "#define NUMA_REGIONS $numa_region" >> $plat_headfile
echo "" >> $plat_headfile
echo "// Cache size in Bytes" >> $plat_headfile
echo "#define CPU_L3CACHE_SIZE $cpu_l3cache_bytes" >> $plat_headfile
echo "#define CPU_L2CACHE_SIZE $cpu_l2cache_bytes" >> $plat_headfile
echo "#define CPU_L1DCACHE_SIZE $cpu_l1d_cache_bytes" >> $plat_headfile
echo "#define CPU_L1IACHE_SIZE $cpu_l1i_cache_bytes" >> $plat_headfile
echo "" >> $plat_headfile
echo "// Main Memory size in Giga Bytes" >> $plat_headfile
echo "#define MAIN_MEM_SIZE $total_mem_GB" >> $plat_headfile

echo "" >> $plat_headfile
echo "#endif // CONFIG_H" >> $plat_headfile

# Display the generated file
echo "$plat_headfile generated with the following content:"
cat $plat_headfile
