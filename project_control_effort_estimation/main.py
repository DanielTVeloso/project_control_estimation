#!/usr/bin/python

from project_control_effort_estimation.modules import model
from project_control_effort_estimation.modules import server

if __name__ == '__main__':
    #if needed train both models
    #df = utils.read_from_storage('storage/df_tf_idf.csv') 
    #model.train_model(df, target='Effort', model = 'all', save_path='storage/models')
    #model.train_model(df, target='Estimated time', model = 'all', save_path='storage/models')

    #run Flask server
    server.run_server()