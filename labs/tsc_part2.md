# Part 2: Image Classification with Custom models
In Part 2, you will find out how to use a custom model with the Image Classification GoogLeNetv1 example from above. Here you will use the same network, but will provide a new model. For this, you will first compile the network again, then quantize a different 32-bit model and execute with the same Python APIs. The model this lab will use is Flowers102, but any custom model trained from GoogLeNetv1 would work.

This part uses the same pyXDNN dir:

```sh
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
```

You will use the xfDNN Tools located in the `/xfdnn_18_04_02/xfdnn_tools/` dir.
```sh
xfdnn_tools/
├── compile
│   ├── codegeneration
│   ├── docs
│   │   ├── caffee.pdf
│   │   ├── DDR.pdf
│   │   ├── networks.pdf
│   │   ├── rules.pyc
│   │   ├── rules.txt
│   │   └── tensorflow.pdf
│   ├── graph
│   ├── memory
│   ├── network
│   ├── notes.sh
│   ├── optimizations
│   ├── README.mxnet
│   ├── README.rst
│   ├── scripts
│   ├── tests
│   │   ├── fullmonty.pyc
│   │   ├── mxnet.pyc
│   │   ├── xfdnn_compiler_inflammable.pyc
│   │   ├── xfdnn_compiler.pyc
│   │   └── xfdnn_compiler_tensorflow.pyc
│   ├── version
│   └── weights
├── models
└── quantize
    ├── quantize_base.pyc
    ├── quantize_caffe.pyc
    ├── quantize.pyc
    └── run_quantize.sh
```

This part will also look at other included models in the `/xfdnn_18_04_02/models/` dir. This directory contains four models: GoogLeNetv1, ResNet50, Flowers102 and Places365. Each model includes the original fp32 model along with pre-quantized 16 and 8-bit versions. In this example, you will recompile and quantize the flowers102 model from the original fp32 model and deploy it using the Python APIs from part 1.

```sh
/xfdnn_18_04_02/models/
├── bvlc_googlenet_without_lrn
├── flowers102
│   ├── fp32
│   │   ├── bvlc_googlenet_without_lrn.caffemodel
│   │   ├── bvlc_googlenet_without_lrn_deploy.prototxt
│   │   ├── ...
│   ├── int16
│   └── int8
├── googlenet_v1
├── places365
└── resnet
```

Ready? Let's begin.

1. Navigate to `/home/centos/xfdnn_18_04_02/caffe/`
	```
	$ ls
	classification.bin  kernelSxdnn_hw_f1_16b.xclbin  run_common.sh         run_places_16b.sh  xdnn_scheduler
	data                kernelSxdnn_hw_f1_8b.xclbin   run_cpu_env.sh        run_resnet_16b.sh  xlnx-docker
	demo                libs                          run_flowers_16b.sh    run_resnet_8b.sh   xlnx-xdnn-f1
	examples            models                        run_googlenet_16b.sh  sdaccel.ini
	execDocker.sh       README                        run_googlenet_8b.sh   start_docker.sh
	```

2. Execute `./start_docker.sh` to enter the application docker.
	```sh
	$ ./start_docker.sh
	/opt#
	```

3. Set XFDNN_ROOT to **/xlnx**
	```
	export XFDNN_ROOT=/xlnx/xfdnn_tools/compile/
	```

4. Navigate to `/xlnx/xfdnn_tools/compile/`
	```
	# cd /xlnx/xfdnn_tools/compile/
	```

5. The next command will execute the GoogLeNet-v1 compiler using a prototxt for caffe. Here, you will pass the GoogLeNetv1 prototxt and the caffemodel from the `models/flowers102` dir.

    **Note**: In step 2 you entered the caffe container, which is needed to run the xfDNN tools. Outside the container, the dir structure is `/home/centos/xfdnn_18_04_02/...`, within the container this maps to `/xlnx/`.

    The compiler version you will use is `xf_dnn_compiler_inflamable.pyc`. The associated parameters are:
    - `[-n,]` - Input prototxt for compiler
    - `[-s,]` - Strategies for compiler (default: all)
    - `[-m,]` - On-chip Memory available in MB (default: 4)
    - `[-i,]` - xfDNN kernel dimension (28 or 56, default: 28)
    - `[-w,--weights WEIGHTS]` -  Caffe trained model (optional)
    - `[-g,]` -  Output commands in XFDNN test and json
    - `[-o,]` -  Output png file of graph read by compiler (optional)

    For the `-n` and `-w` parameters, use the deploy prototxt and caffemodel respectively from the `/flowers102` dir. (see the dir tree at the beginning of part 2.)

    `-s`, `-m`, `-i` are all set to the default values of `all`, `4` and `28` respectively.

    `-g` This is the output of the compiler and is needed for use with the Python APIs in deployment. Assign this file a name and save it back in the `modes/flowers102` directory so it can be located later.

    `-o` This will produce a picture of the graph for reference. Its an optional step. Give this a name as well, in the `/models/flowers102` dir.

    Execute your python command with the above arguments set:
    ```shell
    /xlnx/xfdnn_tools/compile# python tests/xfdnn_compiler_inflammable.pyc \
    -n /xlnx/models/flowers102/fp32/bvlc_googlenet_without_lrn_deploy.prototxt \
    -s all -m 4 -i 28 \
    -g /xlnx/models/flowers102/flowers102_googlenet.cmd \
    -w /xlnx/models/flowers102/fp32/bvlc_googlenet_without_lrn.caffemodel \
    -o /xlnx/models/flowers102/flowers102_graph.png
    ```

    You will receive a long string of console messages while the compiler is running, but on completion you should see `SUCCESS`. If you don't, an error message will explain if you had any problems with your arguments.

    To verify the success of this step, check the `models/flowers102/` dir for the generated files:

    ```sh
    /xlnx/xfdnn_tools/compile# cd /xlnx/models/flowers102/
    /xlnx/models/flowers102# ls
    flowers102_googlenet.cmd  flowers102_googlenet.cmd.json  flowers102_googlenet.cmd.out  flowers102_graph.png  fp32  int16  int8  original.png
    /xlnx/models/flowers102#
    ```

    Confirm the files `flowers102_googlenet.cmd` and `flowers102_graph.png` are there.


6. Quantize the fp32 flowers102 model to int8. Navigate to `/xlnx/xfdnn_tools/quantize/`
  	```sh
    /xlnx/models/flowers102# cd /xlnx/xfdnn_tools/quantize/
    /xlnx/xfdnn_tools/quantize# ls
    quantize.pyc  quantize_base.pyc  quantize_caffe.pyc  run_quantize.sh
    /xlnx/xfdnn_tools/quantize#
  	```

7. In this lab, you will use the `quantize.pyc` tool. The parameters are:
    - `--deploy_model` - This input will be the prototxt for the network/model you want to quantize.
    - `--weights` - This input will the fp32  caffemodel.
    - `--calibration_directory` - This is where the original data set is located, which was used to train the model.
    - `--calibration_size` - This input is the number of images you want to provide for the calibration process.

    `--deploy_model` and `--weights` will be the same from step 5, located in the `models/flowers/` dir.

    The original images required for calibration are located in `/xlnx/imagenet_val`.

    `calibration_size` can be set to a default `8`.

    Execute your python command with the above arguments set:
  	```sh
    /xlnx/xfdnn_tools/quantize# python quantize.pyc \
    --deploy_model /xlnx/models/flowers102/fp32/bvlc_googlenet_without_lrn_deploy.prototxt \
    --weights /xlnx/models/flowers102/fp32/bvlc_googlenet_without_lrn.caffemodel \
    --calibration_directory ../../imagenet_val/ \
    --calibration_size 8
  	```

	If this runs successfully you will see the following at the end of the console messages:
    ```sh
    Passing
    Writing output files to /xlnx/models/flowers102/fp32/bvlc_googlenet_without_lrn_deploy.json...
    Arguments:
    --deploy_model /xlnx/models/flowers102/fp32/bvlc_googlenet_without_lrn_deploy.prototxt
    --weights /xlnx/models/flowers102/fp32/bvlc_googlenet_without_lrn.caffemodel
    --calibration_directory ../../imagenet_val/
    --calibration_size 8
    --calibration_seed None
    --calibration_indices None
    --bitwidths [8,8,8]
    --dims [3,224,224]
    --transpose [2,0,1]
    --channel_swap [2,1,0]
    --raw_scale 255.0
    --mean_value [104.0,117.0,123.0]
    --input_scale 1.0
    ```
    This will recap the quatization run, and will share where the output is located. Output files will be written to `/xlnx/models/flowers102/fp32/bvlc_googlenet_without_lrn_deploy.json.` This JSON file is required to deploy the quantized model.

    It will also store the new weights where the original model was located. For this example that is in the `../fp32/` dir. The new weights data dir will be appended with the suffix `_data`. Verify that `../fp32/bvlc_googlenet_without_lrn.caffemodel_data/` is now created. This will be the path that is provided to the Python APIs for deployment.

    Congrats! You have now compiled and quantized your custom network/model for deployment.

8. Exit the caffe docker now and navigate back to `/pyxdnn/examples/batch_classify/`
    ```sh
    /xlnx/xfdnn_tools/quantize# exit
    exit
    [caffe]$ cd ../pyxdnn/examples/batch_classify/
    ```

9. Open the `lab_part2.sh` script. This is the same Pyhton API from Part 1, but now you will add in the pointers to the outputs generated from the xfDNN tools as well as flower images/labels to test the new model.

10. Set your working dir to where your xfDNN outputs are located (this simplifies references to them). Throughout this lab, it has been  `/home/centos/xfdnn_18_04_02/models/flowers102`.
    ```sh
    # Set Working Dir Path
    export MODELS=/home/centos/xfdnn_18_04_02/models/flowers102
    ```

11. Below shows what you should see next. Included are comments to help provide the right input parameters. The default values are already filled in. The parameters you need, but you didn't generate are:

    - `-outsz`    - For flowers102 the number of categories is `102`
    - `--labels`  - Labels are located here: `../../caffe/data/flowers102/synset_words.txt`
    - `--imagedir`- For test images of flowers, use this dir: `../../caffe/examples/flowers`

    ```sh
    # --------------------------------------------------------------------
    # Part 2: Running GoogleNetv1 with Custom model
    # -------------------------------------------------------------------


    echo " "
    echo "----------------------------------------------------------"
    echo "Image Classification with GoogleNetv1 with Custom 8 bit model"
    echo "----------------------------------------------------------"
    echo " "
    echo " "

    /usr/bin/python batch_classify.py \
    --xclbin $PYXDNN_ROOT/kernel_8b.xclbin \		#8bit 4 kernel binary. Leave as default
    --xlnxnet $MODELS/ \				                #Output of the Compiler ".cmd"
    --fpgaoutsz 1024 \                          #Parameter for final layer. Leave as default
    --outsz  \								                  #Number of Categories
    --datadir $MODELS/ \	                      #Quantized weights data dir
    --labels $MODELS/ \		                      #Text File for Labels
    --xlnxlib $PYXDNN_ROOT/lib/libxblas.so \		#Xilinx Library. Leave as default.
    --imagedir $MODELS/ \			                  #Images dir to classify
    --useblas \								                  #Flag for Blas. Leave as default.
    --xlnxcfg $MODELS/		                      #Quantization data ".json"

    ```


12. When complete, save and exit `lab_part2.sh`. Now execute your run with `sudo ./lab_part2.tsc`. If it runs successfully you should receive the classification results for a number of flowers, which looks like this:
    ```sh
    ---------- Prediction 0 for /home/centos/xfdnn_18_04_02/models/flowers102/../../caffe/examples/flowers/dahlia_03000.jpg
    1.0000 - "pink-yellow dahlia"
    0.0000 - "thorn apple"
    0.0000 - "bearded iris"
    0.0000 - "geranium"
    0.0000 - "petunia"
    ```
    If you receive errors, review your parameters to make sure they are correct. If you can't figure it out, take a look at the `lab_key.sh` script file.





### Congrats! You have compiled and quantized a model and used the Python APIs to deploy it.



[here]: tutorials/launching_instance.md
[compiler]: tutorials/compile.md
[quantizer]: tutorials/quantize.md
[Xilinx ML Suite]: https://github.com/Xilinx/ML-Suite
[Batch Classification example]: https://github.com/Xilinx/ML-Suite/blob/master/pythonexample.md
