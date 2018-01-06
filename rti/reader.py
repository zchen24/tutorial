#!/usr/bin/env python

from time import sleep
import os
import rticonnextdds_connector as rti

qosFile = os.path.dirname(os.path.abspath(__file__)) + "/USER_QOS_PROFILES.reader.xml"
connector = rti.Connector("MyParticipantLibrary::Test",
                          qosFile)

sleep(1)

input = connector.getInput("MySubscriber::MySquareReader")
input.take()
numOfSamples = input.samples.getLength()
print("sample size = " + str(numOfSamples))

for i in range(1, numOfSamples+1):
    id = input.samples.getNumber(i, "sampleId");
    print("sampleid = " + str(id))
 
connector.delete()
