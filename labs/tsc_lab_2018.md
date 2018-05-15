# Xilinx  ML Suite TSC Lab 2018

## Introduction
Welcome to the Xilinx ML Suite TSC Lab. By completing this lab, you will gain valuable hands on experience in using the Xilinx ML Suite. This lab covers how to use Python APIs to deploy the included models, as well as how to compile and quantize custom models with the xfDNN Python tools.

The lab is divided into two parts:

**Part 1**: Uses Python APIs to run the included Image Classification examples with GoogLeNetv1 and pre-quantized 8-bit model.

**Part 2**: Uses the xfDNN tools to compile and quantize a custom model to be used with the Image Classification example in Part 1.

This lab will follow tutorials from the [Xilinx ML Suite][].

## Launching your ML Suite Instance

For this event, each registered participant has been attributed a preconfigured EC2 F1 instance and should have received an email with the following details:

- Account ID
- IAM username
- Link to access a preconfigured EC2 F1 instance

If you have not received that email, please contact an Xilinx event staff member now.

Prior to running the labs, you need to start a pre-configured EC2 F1 instance and connect to it using a ssh client. Once connected, you will be able to run the labs in Parts 1 and 2.

#### Start a preconfigured EC2 F1 instance

1. Open the above mentioned email containing your connection credentials.

2. Click on the link to your preconfigured instance.
    - You will be asked to sign-in before accessing your instance in the AWS EC2 Console.

3. In the AWS sign-in page, enter the **Account ID** which was emailed to you.

4. Click **Next**.

5. Enter the **IAM user name** which was emailed to you.

6. Enter the **Password** provided at the event.

7. Click **Sign In**.
    - You should now be logged in the AWS EC2 Console and you should see an **f1.2xlarge** instance in the **stopped** state.

8. Start the instance by choosing the **Actions** button, then select **Instance State** > **Start**.

    ![Start](imgs/start1.png?)

    - Allow about 10 seconds for the instance to start and to enter the **Running** state.
    - If required, click the **Refresh** icon (![Refresh](imgs/refresh2.png)) in the top-right corner of the EC2 Console to update the instance status information.

9. When the instance is running, find and note the **IPv4 Public IP** address of your instance. This is displayed in EC2 Console next to the instance status.
    - This IP address is used to connect to your instance.

    ![ip](imgs/ipv4.PNG)

#### Connect to your instance using a ssh client

You can now connect to your instance using ssh. You can do this in Linux or Windows as follows:

##### Linux
1. Open a Terminal.

2. Download [tsc_lab_keys.zip][] and save it in your current working directory and unzip.

3. Change the permissions of the **tsc_lab.pem** to **read only**.
    ```sh
    chmod 400 tsc_lab.pem
    ```

4. Execute the following ssh connection command:

    Use the following command if you are behind a Xilinx Firewall:

    For CentOS 7 or later:

    ```sh
    ssh -i tsc_lab.pem -o "ProxyCommand /usr/bin/ncat --proxy-type socks4 --proxy proxy:1080 %h %p" centos@<public-ip>
    ```
    For Older OS versions use:
    ```sh
    ssh -i tsc_lab.pem -o "ProxyCommand /usr/bin/nc -x proxy:1080 %h %p" centos@<public-ipv4-address>
    ```
    Use the following command if you are *not* behind a Xilinx Firewall:
    ```sh
    ssh -i tsc_lab.pem -o centos@<public-ip>
    ```

    **Note**: Replace `<public-ip>` with the **IPv4 Public IP** address for the instance assigned to you.

5. Type `yes` when prompted to connect.
    ```sh
    RSA key fingerprint is f9:f4:05:da:40:99:00:9d:1e:8f:95:de:89:6a:3f:cd.
    Are you sure you want to continue connecting (yes/no)? yes
    ```
You should now be connected to the remote F1 instance running Centos 7.

##### Windows
1. Download [tsc_lab_keys.zip][] and save it in your current working directory and unzip.
2. Launch PuTTY. If required, [download from here][].
3. Enter the following PuTTY session configuration details:

* **Host Name**: Enter **centos@** followed by your **IPv4 Public Address**.

* **Connection Type**: Set to **SSH**

* **Port**: Leave at the default **22**

   See the following figure for reference.

    ![](imgs/putty_ip.PNG)

4. Under **Connection**, click the `+` next to **SSH** to expand, and click **Auth**.
5. Click **Browse** and provide the path to the **tsc_lab.ppk** file you downloaded earlier. Leave everything else as default. See the following figure for reference.

    ![](imgs/putty_auth.PNG)

6. If you are behind the Xilinx Firewall, click **Proxy** on the left and select **HTTP** as the **Proxy type**. Leave everything else as default. See the following figure for reference.

    ![](imgs/putty_proxy.PNG)

7. Click **Open** at the bottom of the window. The following pop-up window will display. Click **Yes** to proceed. See the following figure for reference.

    ![](imgs/putty_rsa.PNG)

You should now be connected to the remote F1 instance running Centos 7.



## [Start Part 1: Image Classification with Python APIs][]



[here]: tutorials/launching_instance.md
[compiler]: tutorials/compile.md
[quantizer]: tutorials/quantize.md
[Xilinx ML Suite]: https://github.com/Xilinx/ML-Suite
[Batch Classification example]: https://github.com/Xilinx/ML-Suite/blob/master/pythonexample.md
[Start Part 1: Image Classification with Python APIs]: tsc_part1.md
[tsc_lab_keys.zip]: https://github.com/kamranjk/ML-Development-Stack-From-Xilinx/raw/tsc2018/labs/keys/tsc_lab_keys.zip
[download from here]: https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html
