	BwI?W??@BwI?W??@!BwI?W??@	?BHC?X@?BHC?X@!?BHC?X@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6BwI?W??@0.Ui?@15?b???K@A??????I???E??Yc&Q/??@*	?j???lA2j
3Iterator::Model::MaxIntraOpParallelism::MapAndBatch??????@!ی???X@)??????@1ی???X@:Preprocessing2s
<Iterator::Model::MaxIntraOpParallelism::MapAndBatch::ShuffleFaE|??!???6O?)FaE|??1???6O?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism&?v????@!?????X@)n½2oս?14?V?/I?:Preprocessing2F
Iterator::Model{Cr???@!?9?d??X@)????y}?1????S??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 99.6% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9?BHC?X@I????????Q4??] ???Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	0.Ui?@0.Ui?@!0.Ui?@      ??!       "	5?b???K@5?b???K@!5?b???K@*      ??!       2	????????????!??????:	???E?????E??!???E??B      ??!       J	c&Q/??@c&Q/??@!c&Q/??@R      ??!       Z	c&Q/??@c&Q/??@!c&Q/??@b      ??!       JGPUY?BHC?X@b q????????y4??] ???