// Spark imports
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.log4j.Logger
import org.apache.log4j.Level

import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg._
import org.apache.spark.sql.types._
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions.col
import scala.language.implicitConversions

object PartA_draft2 {
	def main(args: Array[String]): Unit ={
	
		val startTimeMillis = System.currentTimeMillis()

		Logger.getLogger("org").setLevel(Level.OFF)
		Logger.getLogger("akka").setLevel(Level.OFF)
		val conf = new SparkConf().setAppName("BigData")
		val sc = new SparkContext(conf)
		val sqlContext = new SQLContext(sc)

		val nodeInfo_temp = sqlContext.read.format("csv").option("header", "false").option("delimiter", ";").load("/home/user/IdeaProjects/BigDataProjectAttempt1/target/scala-2.11/node_information.csv")
		val nodeInfo = nodeInfo_temp.toDF("id", "year", "title", "authors", "journal", "abstract")
		val nodes = nodeInfo.drop("authors").drop("journal")

		// UDFs
		val eucDisUdf = udf((v1: Vector, v2: Vector) => Vectors.sqdist(v1, v2))
		val yearDiffUdf = udf((y1: String, y2: String) => Math.abs(y1.toInt - y2.toInt))
		val toVecUdf = udf((v1: Double, v2: Double, v3: Int) => Vectors.dense(v1, v2, v3))
		val labelToDoubleUdf = udf((v: String) => v.toDouble)

///////////////////////////////////////////////////////////// Training data ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		val edges_temp = sqlContext.read.format("csv").option("header", "false").option("delimiter", ",").load("/home/user/IdeaProjects/BigDataProjectAttempt1/target/scala-2.11/training_set.csv")
		val edges = edges_temp.toDF("source", "target", "label")

///////////////////////////////////////////////////////////// Join Dataframes ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		println("Started joining Dataframes...")

		val join_source = nodes.join(edges, nodes("id") === edges("source"), "fullouter")
		val join_source_2 = join_source.drop("source")

		val join_target = nodes.join(edges, nodes("id") === edges("target"), "fullouter")
		val join_target_2 = join_target.drop("target")

		val joined1 = join_source_2.toDF("source", "year1", "title1", "abstract1", "target", "label")
		val joined2 = join_target_2.toDF("target", "year2", "title2", "abstract2", "source", "label")

		val result1 = joined1.join(nodes, joined1("target") === nodes("id"))
		val result1_temp = result1.drop("source").drop("target").drop("id")
		val data1 = result1_temp.toDF("year1", "title1", "abstract1", "label", "year2", "title2", "abstract2")

		val result2 = joined2.join(nodes, joined2("source") === nodes("id"))
		val result2_temp = result2.drop("source").drop("target").drop("id")
		val data2 = result2_temp.toDF("year2", "title2", "abstract2", "label", "year1", "title1", "abstract1")

		println("Dataframes joined!")
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////		

///////////////////////////////////////////////////////////// Feature extraction ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		println("Started the feature extraction process...")

		// Tokenization
		val tokenizer_abstract1 = new Tokenizer().setInputCol("abstract1").setOutputCol("abstract1_tokens")
		val data1_1 = tokenizer_abstract1.transform(data1)
		data1_1.show
		val tokenizer_abstract2 = new Tokenizer().setInputCol("abstract2").setOutputCol("abstract2_tokens")
		val data1_2 = tokenizer_abstract2.transform(data1_1)
		data1_2.show
		val tokenizer_title1 = new Tokenizer().setInputCol("title1").setOutputCol("title1_tokens")
		val data1_3 = tokenizer_title1.transform(data1_2)
		data1_3.show
		val tokenizer_title2 = new Tokenizer().setInputCol("title2").setOutputCol("title2_tokens")
		val data1_4 = tokenizer_title2.transform(data1_3)
		data1_4.show
		val data = data1_4.drop("title1").drop("title2").drop("abstract1").drop("abstract2")
		data.show

		// Vectorization
		val vocabSize = 1000
		val cvModel_abstract1: CountVectorizerModel = new CountVectorizer().setInputCol("abstract1_tokens").setOutputCol("abstract1_vectors").setVocabSize(vocabSize).setMinDF(10).fit(data)
		//val isNoneZeroVector = udf({v: Vector => v.numNonzeros > 0}, DataTypes.BooleanType)
		//val vectorizedDf = cvModel.transform(data).filter(isNoneZeroVector(col("features")))
		val data_1 = cvModel_abstract1.transform(data)
		data_1.show
		val cvModel_abstract2: CountVectorizerModel = new CountVectorizer().setInputCol("abstract2_tokens").setOutputCol("abstract2_vectors").setVocabSize(vocabSize).setMinDF(10).fit(data_1)
		val data_2 = cvModel_abstract2.transform(data_1)
		data_2.show
		val cvModel_title1: CountVectorizerModel = new CountVectorizer().setInputCol("title1_tokens").setOutputCol("title1_vectors").setVocabSize(vocabSize).setMinDF(10).fit(data_2)
		val data_3 = cvModel_title1.transform(data_2)
		data_3.show
		val cvModel_title2: CountVectorizerModel = new CountVectorizer().setInputCol("title2_tokens").setOutputCol("title2_vectors").setVocabSize(vocabSize).setMinDF(10).fit(data_3)
		val data_4 = cvModel_title2.transform(data_3)
		data_4.show

		// Finding distances between vectors
		val data_features_temp = data_4.drop("abstract1_tokens").drop("abstract2_tokens").drop("title1_tokens").drop("title2_tokens")
		data_features_temp.show
		val feature_vectors_temp_1 = data_features_temp.withColumn("abstract_dist", eucDisUdf(col("abstract1_vectors"), col("abstract2_vectors")))
		feature_vectors_temp_1.show
		val feature_vectors_temp_2 = feature_vectors_temp_1.withColumn("title_dist", eucDisUdf(col("title1_vectors"), col("title2_vectors")))
		feature_vectors_temp_2.show
		val feature_vectors_temp_3 = feature_vectors_temp_2.withColumn("year_diff", yearDiffUdf(col("year1"), col("year2")))
		feature_vectors_temp_3.show
		val feature_vectors_temp_4 = feature_vectors_temp_3.drop("year1").drop("year2").drop("abstract1_vectors").drop("abstract2_vectors").drop("title1_vectors").drop("title2_vectors")
		feature_vectors_temp_4.show
		val feature_vectors_temp_5 = feature_vectors_temp_4.withColumn("features", toVecUdf(col("abstract_dist"), col("title_dist"), col("year_diff")))
		feature_vectors_temp_5.show
		val feature_vectors_temp_6 = feature_vectors_temp_5.withColumn("label_new", labelToDoubleUdf(col("label")))
		feature_vectors_temp_6.show
		val feature_vectors_temp_7 = feature_vectors_temp_6.drop("label").drop("abstract_dist").drop("title_dist").drop("year_diff")
		feature_vectors_temp_7.show
		val feature_vectors = feature_vectors_temp_7.toDF("features", "label")
		feature_vectors.show

		println("Feature extraction process finished!")
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////// End Training data ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////// Test data ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		val test_edges_temp = sqlContext.read.format("csv").option("header", "false").option("delimiter", ",").load("/home/user/IdeaProjects/BigDataProjectAttempt1/target/scala-2.11/testing_set.csv")
		val test_edges = test_edges_temp.toDF("source", "target", "label")

///////////////////////////////////////////////////////////// Join Dataframes ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		println("Started joining testing Dataframes...")

		val test_join_source = nodes.join(test_edges, nodes("id") === test_edges("source"), "fullouter")
		val test_join_source_2 = test_join_source.drop("source")

		val test_join_target = nodes.join(test_edges, nodes("id") === test_edges("target"), "fullouter")
		val test_join_target_2 = test_join_target.drop("target")

		val test_joined1 = test_join_source_2.toDF("source", "year1", "title1", "abstract1", "target", "label")
		val test_joined2 = test_join_target_2.toDF("target", "year2", "title2", "abstract2", "source", "label")

		val test_result1 = test_joined1.join(nodes, test_joined1("target") === nodes("id"))
		val test_result1_temp = test_result1.drop("source").drop("target").drop("id")
		val test_data1 = test_result1_temp.toDF("year1", "title1", "abstract1", "label", "year2", "title2", "abstract2")

		val test_result2 = test_joined2.join(nodes, test_joined2("source") === nodes("id"))
		val test_result2_temp = test_result2.drop("source").drop("target").drop("id")
		val test_data2 = test_result2_temp.toDF("year2", "title2", "abstract2", "label", "year1", "title1", "abstract1")

		println("Testing Dataframes joined!")

///////////////////////////////////////////////////////////// Feature extraction ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		println("Started the feature extraction process of the testing data...")

		// Tokenization
		val test_tokenizer_abstract1 = new Tokenizer().setInputCol("abstract1").setOutputCol("abstract1_tokens")
		val test_data1_1 = test_tokenizer_abstract1.transform(test_data1)
		test_data1_1.show
		val test_tokenizer_abstract2 = new Tokenizer().setInputCol("abstract2").setOutputCol("abstract2_tokens")
		val test_data1_2 = test_tokenizer_abstract2.transform(test_data1_1)
		test_data1_2.show
		val test_tokenizer_title1 = new Tokenizer().setInputCol("title1").setOutputCol("title1_tokens")
		val test_data1_3 = test_tokenizer_title1.transform(test_data1_2)
		test_data1_3.show
		val test_tokenizer_title2 = new Tokenizer().setInputCol("title2").setOutputCol("title2_tokens")
		val test_data1_4 = test_tokenizer_title2.transform(test_data1_3)
		test_data1_4.show
		val test_data = test_data1_4.drop("title1").drop("title2").drop("abstract1").drop("abstract2")
		test_data.show

		// Vectorization
		val test_vocabSize = 1000
		val test_cvModel_abstract1: CountVectorizerModel = new CountVectorizer().setInputCol("abstract1_tokens").setOutputCol("abstract1_vectors").setVocabSize(test_vocabSize).setMinDF(10).fit(test_data)
		//val isNoneZeroVector = udf({v: Vector => v.numNonzeros > 0}, DataTypes.BooleanType)
		//val vectorizedDf = cvModel.transform(data).filter(isNoneZeroVector(col("features")))
		val test_data_1 = test_cvModel_abstract1.transform(test_data)
		test_data_1.show
		val test_cvModel_abstract2: CountVectorizerModel = new CountVectorizer().setInputCol("abstract2_tokens").setOutputCol("abstract2_vectors").setVocabSize(test_vocabSize).setMinDF(10).fit(test_data_1)
		val test_data_2 = test_cvModel_abstract2.transform(test_data_1)
		test_data_2.show
		val test_cvModel_title1: CountVectorizerModel = new CountVectorizer().setInputCol("title1_tokens").setOutputCol("title1_vectors").setVocabSize(test_vocabSize).setMinDF(10).fit(test_data_2)
		val test_data_3 = test_cvModel_title1.transform(test_data_2)
		test_data_3.show
		val test_cvModel_title2: CountVectorizerModel = new CountVectorizer().setInputCol("title2_tokens").setOutputCol("title2_vectors").setVocabSize(test_vocabSize).setMinDF(10).fit(test_data_3)
		val test_data_4 = test_cvModel_title2.transform(test_data_3)
		test_data_4.show

		// Finding distances between vectors
		val test_data_features_temp = test_data_4.drop("abstract1_tokens").drop("abstract2_tokens").drop("title1_tokens").drop("title2_tokens")
		test_data_features_temp.show
		val test_feature_vectors_temp_1 = test_data_features_temp.withColumn("abstract_dist", eucDisUdf(col("abstract1_vectors"), col("abstract2_vectors")))
		test_feature_vectors_temp_1.show
		val test_feature_vectors_temp_2 = test_feature_vectors_temp_1.withColumn("title_dist", eucDisUdf(col("title1_vectors"), col("title2_vectors")))
		test_feature_vectors_temp_2.show
		val test_feature_vectors_temp_3 = test_feature_vectors_temp_2.withColumn("year_diff", yearDiffUdf(col("year1"), col("year2")))
		test_feature_vectors_temp_3.show
		val test_feature_vectors_temp_4 = test_feature_vectors_temp_3.drop("year1").drop("year2").drop("abstract1_vectors").drop("abstract2_vectors").drop("title1_vectors").drop("title2_vectors")
		test_feature_vectors_temp_4.show
		val test_feature_vectors_temp_5 = test_feature_vectors_temp_4.withColumn("features", toVecUdf(col("abstract_dist"), col("title_dist"), col("year_diff")))
		test_feature_vectors_temp_5.show
		val test_feature_vectors_temp_6 = test_feature_vectors_temp_5.withColumn("label_new", labelToDoubleUdf(col("label")))
		test_feature_vectors_temp_6.show
		val test_feature_vectors_temp_7 = test_feature_vectors_temp_6.drop("label").drop("abstract_dist").drop("title_dist").drop("year_diff")
		test_feature_vectors_temp_7.show
		val test_feature_vectors = test_feature_vectors_temp_7.toDF("features", "label")
		test_feature_vectors.show

		println("Feature extraction process of the testing data finished!")
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////// End Test data ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////// Classification ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		println("Classification...")
		val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
		println("Training...")
		val lrModel = lr.fit(feature_vectors)
		
		println("Measuring training error...")
		val predictions = lrModel.transform(feature_vectors)
		val scoreAndLabels = predictions.select("label", "probability").rdd.map(row => (row.getAs[Vector]("probability")(1), row.getAs[Double]("label")))
		val metrics = new BinaryClassificationMetrics(scoreAndLabels)
		val f1Score = metrics.fMeasureByThreshold
		f1Score.foreach { case (t, f) => println(s"Threshold: $t, F-score: $f, Beta = 1")}

		println("Measuring testing error...")
		val test_predictions = lrModel.transform(test_feature_vectors)
		val test_scoreAndLabels = test_predictions.select("label", "probability").rdd.map(row => (row.getAs[Vector]("probability")(1), row.getAs[Double]("label")))
		val test_metrics = new BinaryClassificationMetrics(test_scoreAndLabels)
		val test_f1Score = test_metrics.fMeasureByThreshold
		test_f1Score.foreach { case (t, f) => println(s"Threshold: $t, F-score: $f, Beta = 1")}

		// Save test predictions
		val selected = test_predictions.select("prediction")
		selected.coalesce(1).write.csv("/home/user/IdeaProjects/BigDataProjectAttempt1/target/scala-2.11/predictions.csv")

		val endTimeMillis = System.currentTimeMillis()
		val durationSeconds = (endTimeMillis - startTimeMillis) / 1000
		println("Elapsed time in seconds: " + durationSeconds)
		
	}
	
}

