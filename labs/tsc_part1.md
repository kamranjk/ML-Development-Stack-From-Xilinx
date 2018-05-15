# Part 1 - Image Classification with Python APIs
In Part 1, you will run through the [Batch Classification example][], using the xfDNN Python APIs.

Directory overview of this tutorial:

<pre>
pyxdnn/
└── examples
    ├── batch_classify
    │   ├── batch_classify.py					
    │   ├── images
    │   ├── images_224
    │   ├── run.sh
    │   ├── tsc.sh											
    │   └── synset_words.txt
    ├── common
    └── multinet
        ├── mytest.json
        ├── run.sh
        ├── synset_words.txt
        └── test_classify_async_multinet.py
</pre>

1. Navigate to: `/home/centos/xfdnn_18_04_02/pyxdnn/examples/batch_classify/`

	```bash
	$ cd xfdnn_18_04_02/pyxdnn/examples/batch_classify/
	$ ls
	batch_classify.py  images  images_224  run.sh  synset_words.txt  lab_part1.sh lab_part2.sh
	```

2. Open `lab_part1.sh` with a text editor of your choice. These labs use nano as it's a simple editor. When using nano, remember to use `sudo` to launch it, so you can save your edits. Execute `sudo nano lab_part1.sh` to view the contents of the script.

	The top of the file includes environment and setup commands. No changes are required here.  
	```sh
	#!/usr/bin/env bash

	if [ "$EUID" -ne 0 ]
	  then echo "Please run as root"
	  exit
	fi

	export PYXDNN_ROOT=../common
	export LD_LIBRARY_PATH=$PYXDNN_ROOT/runtime/lib/x86_64:$PYXDNN_ROOT/lib
	export XILINX_OPENCL=$PYXDNN_ROOT
	source $PYXDNN_ROOT/setup.sh
	```
	Beginning on line 14, you will see the python command and parameters required to run the Image Classification example.
	```sh
	# --------------------------------------------------------------------
	# Part 1: Running GoogleNetv1 with Standard 1000 class model
	# -------------------------------------------------------------------

	echo "Image Classification with GoogleNetv1 with 8 bit model"

	/usr/bin/python batch_classify.py \
	--xclbin $PYXDNN_ROOT/kernel_8b.xclbin \
	--xlnxnet $PYXDNN_ROOT/xdnn_scheduler/googlenet.fpgaaddr.64.txt \
	--fpgaoutsz 1024 \
	--datadir $PYXDNN_ROOT/data_googlenet_v1 \
	--labels synset_words.txt \
	--xlnxlib $PYXDNN_ROOT/lib/libxblas.so \
	--imagedir images_224 \
	--useblas \
	--xlnxcfg $PYXDNN_ROOT/xdnn_scheduler/googlenet_quantized.json

	```

	This example runs eight images through the Python API in either **batch** or **streaming** mode.

	The `lab_part1.sh` example calls the API, and takes the following parameters:
	batch_classify.py
	- `--xclbin` 		- Defines which FPGA binary to use. By Default, leave this set to the binary given in the example
	- `--xlnxcfg` 	- FPGA config file
	- `--xlnxnet` 	- FPGA instructions generated by the xfDNN Compiler for the network
	- `--fpgaoutsz`	- Size of the FPGA output blob
	- `--datadir`		- Path to data files to run for the network (weights)
	- `--labels`		- Result -> labels translation file (typically a text File)
	- `--xlnxlib`		- FPGA xfDNN lib
	- `--imagedir`	- Directory containing image files to classify
	- `--useblas`		- Use BLAS-optimized functions (requires xfDNN lib compiled with BLAS)

	Exit the file once you have become familiar with the script and its components.

3. Execute the example by running `sudo ./lab_part1.sh`.

	```sh
	$ sudo ./lab_part1.sh
	Image Classification with GoogleNetv1 with 8 bit model
	[XBLAS] # kernels: 1
	[XDNN] using custom DDR banks 0,2,1,1
	Device/Slot[0] (/dev/xdma0, 0:0:1d.0)
	xclProbe found 1 FPGA slots with XDMA driver running
	[time] loadImageBlobFromFile/OpenCV (4.15 ms):
	[time] loadImageBlobFromFile/OpenCV (2.13 ms):
	[time] loadImageBlobFromFile/OpenCV (1.91 ms):
	[time] loadImageBlobFromFile/OpenCV (1.45 ms):
	[time] loadImageBlobFromFile/OpenCV (1.56 ms):
	[time] loadImageBlobFromFile/OpenCV (1.57 ms):
	[time] loadImageBlobFromFile/OpenCV (1.60 ms):
	[time] loadImageBlobFromFile/OpenCV (2.19 ms):
	[time] prepareImages (24.43 ms):
	CL_PLATFORM_VENDOR Xilinx
	CL_PLATFORM_NAME Xilinx
	CL_DEVICE_0: 0x27a6e30
	CL_DEVICES_FOUND 1, using 0
	loading ../common/kernel_8b.xclbin
	[XBLAS] kernel0: kernelSxdnn_0
	Loading weights/bias/quant_params to FPGA...
	WARNING: unaligned host pointer detected, this leads to extra memcpy
	WARNING: unaligned host pointer detected, this leads to extra memcpy
	WARNING: unaligned host pointer detected, this leads to extra memcpy
	WARNING: unaligned host pointer detected, this leads to extra memcpy
	WARNING: unaligned host pointer detected, this leads to extra memcpy
	WARNING: unaligned host pointer detected, this leads to extra memcpy
	WARNING: unaligned host pointer detected, this leads to extra memcpy
	WARNING: unaligned host pointer detected, this leads to extra memcpy
	[XDNN] FPGA metrics (0/0/0)
	[XDNN]   write_to_fpga  : 0.17 ms
	[XDNN]   exec_xdnn      : 39.57 ms
	[XDNN]   read_from_fpga : 0.06 ms
	[XDNN] FPGA metrics (0/1/0)
	[XDNN]   write_to_fpga  : 0.15 ms
	[XDNN]   exec_xdnn      : 39.42 ms
	[XDNN]   read_from_fpga : 0.06 ms
	[XDNN] FPGA metrics (0/2/0)
	[XDNN]   write_to_fpga  : 0.15 ms
	[XDNN]   exec_xdnn      : 38.95 ms
	[XDNN]   read_from_fpga : 0.08 ms
	[XDNN] FPGA metrics (0/3/0)
	[XDNN]   write_to_fpga  : 0.17 ms
	[XDNN]   exec_xdnn      : 43.37 ms
	[XDNN]   read_from_fpga : 0.05 ms
	[time] FPGA xdnn execute (580.57 ms):
	[time] FC (2.00 ms):

	---------- Prediction 0 for images_224/cat gray.jpg ----------
	0.8056 - "n02123394 Persian cat"
	0.1379 - "n02127052 lynx, catamount"
	0.0165 - "n02326432 hare"
	0.0130 - "n02120079 Arctic fox, white fox, Alopex lagopus"
	0.0098 - "n02123159 tiger cat"

	---------- Prediction 1 for images_224/cat.jpg ----------
	0.9730 - "n02123159 tiger cat"
	0.0164 - "n02124075 Egyptian cat"
	0.0036 - "n02123045 tabby, tabby cat"
	0.0020 - "n02127052 lynx, catamount"
	0.0014 - "n02119789 kit fox, Vulpes macrotis"

	---------- Prediction 2 for images_224/ILSVRC2012_val_00003225.JPEG ----------
	0.9033 - "n02422106 hartebeest"
	0.0780 - "n02423022 gazelle"
	0.0183 - "n02422699 impala, Aepyceros melampus"
	0.0001 - "n02408429 water buffalo, water ox, Asiatic buffalo, Bubalus bubalis"
	0.0001 - "n02437616 llama"

	---------- Prediction 3 for images_224/ILSVRC2012_val_00001172.JPEG ----------
	0.4223 - "n02114548 white wolf, Arctic wolf, Canis lupus tundrarum"
	0.3238 - "n02109961 Eskimo dog, husky"
	0.2243 - "n02114367 timber wolf, grey wolf, gray wolf, Canis lupus"
	0.0200 - "n02110063 malamute, malemute, Alaskan malamute"
	0.0073 - "n02110185 Siberian husky"

	---------- Prediction 4 for images_224/fish-bike.jpg ----------
	0.5581 - "n02797295 barrow, garden cart, lawn cart, wheelbarrow"
	0.3857 - "n04482393 tricycle, trike, velocipede"
	0.0271 - "n02835271 bicycle-built-for-two, tandem bicycle, tandem"
	0.0127 - "n04258138 solar dish, solar collector, solar furnace"
	0.0020 - "n03599486 jinrikisha, ricksha, rickshaw"

	---------- Prediction 5 for images_224/cat_gray.jpg ----------
	0.8056 - "n02123394 Persian cat"
	0.1379 - "n02127052 lynx, catamount"
	0.0165 - "n02326432 hare"
	0.0130 - "n02120079 Arctic fox, white fox, Alopex lagopus"
	0.0098 - "n02123159 tiger cat"

	---------- Prediction 6 for images_224/cat gray.jpg ----------
	0.8056 - "n02123394 Persian cat"
	0.1379 - "n02127052 lynx, catamount"
	0.0165 - "n02326432 hare"
	0.0130 - "n02120079 Arctic fox, white fox, Alopex lagopus"
	0.0098 - "n02123159 tiger cat"

	---------- Prediction 7 for images_224/cat.jpg ----------
	0.9730 - "n02123159 tiger cat"
	0.0164 - "n02124075 Egyptian cat"
	0.0036 - "n02123045 tabby, tabby cat"
	0.0020 - "n02127052 lynx, catamount"
	0.0014 - "n02119789 kit fox, Vulpes macrotis"

	Num processed: 8

	[time] Total loop (9748.08 ms):

	```
4. Analyze the performance results. The `[time] loadImageBlobFromFile/OpenCV (4.15 ms):` messages show the time it takes to format the images on the cpu in preparation for Image Classification. This is a one-time operation, performed at the beginning of the run.

	The next important read out is the processing time on the xDNN kernel. Look for the following message:
	```sh
	[XDNN] FPGA metrics (0/0/0)
	[XDNN]   write_to_fpga  : 0.17 ms
	[XDNN]   exec_xdnn      : 39.57 ms
	[XDNN]   read_from_fpga : 0.06 ms
	```
	This shows the write, execution and read times from each of the four xDNN kernels on the FPGA. By default, this example runs eight images (two per kernel). Because each kernel needs to be initialized for the network/model to be run, it has a high processing time. Feeding more images into it will show the **real** processing time. For example, if you change this to run more images, you will see the following results:
	```sh
	[XDNN] FPGA metrics (0/0/0)
	[XDNN]   write_to_fpga  : 0.21 ms
	[XDNN]   exec_xdnn      : 5.27 ms
	[XDNN]   read_from_fpga : 0.06 ms
	```

## [On to Part 2: Image Classification with Custom models][]


  [here]: tutorials/launching_instance.md
  [compiler]: tutorials/compile.md
  [quantizer]: tutorials/quantize.md
  [Xilinx ML Suite]: https://github.com/Xilinx/ML-Suite
  [Batch Classification example]: https://github.com/Xilinx/ML-Suite/blob/master/pythonexample.md
  [On to Part 2: Image Classification with Custom models]: tsc_part2.md