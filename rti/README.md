RTI Hello World
=============


## Dependencies 

The example is generated using RTI DDS 5.3.0 eval version. 

## Install rti connector

https://github.com/rticommunity/rticonnextdds-connector

## Generate C++ Code & Build

```bash
# generate example code from idl
path/to/rti_connext_dds-5.3.0/bin/rtiddsgen -language C++ -example x64Linux3gcc5.4.0 HelloWorldDefs.idl

# build 
make -f makefile_HelloWorldDefs_x64Linux3gcc5.4.0
```


## Run  

* Setup license by having ```RTI_LICENSE_FILE``` environment variable point to a ```rti_license.dat``` file.
* Setup QOS file, ```NDDS_QOS_PROFILES``` point to ```USER_QOS_PROFILES.qos.xml```
* Run C++ publisher 

```bash
# run
./objs/x64Linux3gcc5.4.0/HelloWorldDefs_publisher
```

* Run Python reader

```bash
python ./reader
```