from cnnClassifier import logger
from cnnClassifier.pipeline.stages_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stages_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from cnnClassifier.pipeline.stages_03_training import ModelTrainingPipeline


STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f">>>> stage {STAGE_NAME} started successfully <<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>> stage {STAGE_NAME} completed successfully <<<<<\n\nX===================X")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Prepare Base Model"


try:
    logger.info(f">>>> stage {STAGE_NAME} started successfully <<<<<")
    prepare_base_model = PrepareBaseModelTrainingPipeline()
    prepare_base_model.main()
    logger.info(f">>>> stage {STAGE_NAME} completed successfully <<<<<\n\nX===================X")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Training"


try:
    logger.info(f"****************************************************************")
    logger.info(f">>>> stage {STAGE_NAME} started successfully <<<<<")
    model_trainer = ModelTrainingPipeline()
    model_trainer.main()
    logger.info(f">>>> stage {STAGE_NAME} completed successfully <<<<<\n\nX===================X")
except Exception as e:
    logger.exception(e)
    raise e