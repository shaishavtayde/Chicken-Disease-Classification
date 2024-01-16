from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.data_ingestion import DataIngestion
from cnnClassifier import logger

STAGE_NAME = "data_ingestion"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_files()
        data_ingestion.extract_zip_files()

if __name__ == "__main__":
    try:
        logger.info(f">>>> stage {STAGE_NAME} started successfully <<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>> stage {STAGE_NAME} completed successfully <<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
