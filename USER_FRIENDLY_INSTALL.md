# TRAJECTA — Installation Guide

This guide will walk you through the complete installation of **Trajecta** step by step. No technical background is required — just follow the instructions carefully and you will be ready to go.

---

## Step 1 — Download and Install Trajecta

1. Go to the [Trajecta GitHub page](https://github.com/ArcheoHacker1501/trajecta) and navigate to the **Releases** section (on the right side of the page).
2. Download the installer file. It will be named something like `trajecta-vx.x.x.exe` (where `x.x.x` is the version number).
3. Run the installer by double-clicking on it.
4. Follow the on-screen instructions and complete the installation.

> **⚠️ Important:** Take note of the installation directory (the folder where Trajecta is being installed). This folder will contain `trajecta.exe`, the main application you will use later. You will need this path in the following steps.

---

## Step 2 — Install GDAL (Required)

Before you can run `trajecta.exe`, you need to install **GDAL**, a set of essential geospatial libraries. Without GDAL, Trajecta will not be able to function.

### 2.1 — Download the OSGeo4W Installer

1. Go to [https://trac.osgeo.org/osgeo4w](https://trac.osgeo.org/osgeo4w).
2. Download the **OSGeo4W network installer**.

### 2.2 — Run the OSGeo4W Installer

1. Launch the installer you just downloaded.
2. Select **Advanced Install** (this should be the default option) and click **Next**.
3. Choose **Install from Internet** and click **Next**.
4. Choose the **root installation directory**. This is the main folder where GDAL and related tools will be installed. You can also create a desktop shortcut if you wish.

> **⚠️ Important:** Write down (or copy) this directory path. You might need it later.

5. Choose the **local package directory**. This is a temporary folder where downloaded files are stored during installation.

> **⚠️ Important:** Write down (or copy) this directory path as well.

6. Select your **internet connection type**. The default setting should work fine — just click **Next**.
7. Select one of the available **download sites** from the list and click **Next**.

### 2.3 — Select the Packages to Install

When the package selection screen appears, you will see a list of categories. For each category, there is a label that says "Default" — clicking on it will cycle through options like "Install", "Uninstall", etc.

Configure the categories as follows:

| Category | Action |
|---|---|
| **Commandline_Utilities** | Click on "Default" until it changes to **Install** (install all) |
| **Desktop** | You can skip this (leave as Default). However, if you do not have **QGIS** installed on your computer, it is recommended to install it here by expanding the Desktop category and clicking "Install" next to `qgis: QGIS Desktop`. QGIS is a free GIS application that is essential for visualizing Trajecta's results and for creating the input files needed for the analysis. |
| **Libs** | Click on "Default" until it changes to **Install** (install all) |
| **Web** | Leave as **Default** (nothing will be installed) |

8. Click **Next** to proceed.
9. If a window appears saying that some **dependencies are not satisfied**, you do **not** need to install the suggested additional packages. Uncheck the box at the bottom-left corner of the window, then click **Next**.
10. Accept all license agreements and continue the installation. Wait for all packages to download and install. This may take several minutes depending on your internet connection.
11. Once the installation is complete, close the installer.

### 2.4 — Add GDAL to the System PATH

For Trajecta to find and use the GDAL libraries, you need to add the GDAL folder to your system's **PATH** environment variable. This tells Windows where to look for the files that Trajecta needs.

Follow these steps:

1. In the **Windows search bar** (bottom-left of your screen), type: `Edit the system environment variables` and open the result.
2. In the window that appears, click the **Environment Variables** button (bottom-right).
3. In the new panel, look at the **lower section** labeled **System variables**. Scroll through the list until you find a variable called **Path**. Select it and click **Edit**.
4. In the new window, click **New** and paste the following path:

```
C:\...\OSGeo4W\bin
```

Replace `C:\...\OSGeo4W` with the actual root installation directory you noted during step 2.2. Make sure the path ends with `\bin`.

For example, if you installed OSGeo4W in `C:\OSGeo4W`, the path you need to add is:

```
C:\OSGeo4W\bin
```

> **⚠️ Important:** Do not forget to include `\bin` at the end of the path!

5. Click **OK** three times to close all windows.

GDAL is now installed and configured. Trajecta will be able to access it.

---

## Step 3 — Install NVIDIA GPU Support (Optional)

> **This step is only required if you have a compatible NVIDIA GPU installed in your computer.** If you do not have an NVIDIA GPU, or if you are unsure, you can safely skip this entire section. Trajecta will still work using your CPU. However, you will not be able to use GPU acceleration for FETE analysis — only CPU parallelization will be available.

### 3.1 — Install NVIDIA Drivers

1. Go to [https://www.nvidia.com/en-us/drivers/](https://www.nvidia.com/en-us/drivers/).
2. Under **Get Automatic Driver Updates**, select either **Gamers/Creators** or **Professionals/Workstation Users** (depending on your GPU type).
3. Download the application and follow the installation wizard.
4. Once installed, open the application and **update your GPU driver** to the latest version.

### 3.2 — Install CUDA Toolkit 11.0+

1. Go to [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads).
2. Select the options that match your system:
   - **Operating System:** Windows
   - **Architecture:** x86_64
   - **Version:** Select your Windows version (10 or 11)
   - **Installer Type:** Select `exe (network)`
3. Download the CUDA Toolkit installer and run it.
4. Follow the installation wizard to complete the setup.

---

## You Are Ready!

Once you have completed all the steps above, you can launch **`trajecta.exe`** from the installation directory you noted in Step 1.

Trajecta should now be fully functional. Enjoy your spatial analysis!
