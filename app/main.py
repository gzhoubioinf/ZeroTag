import sys, os, yaml

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from example_tagging import example_tagging_app
# from ml_prediction import app_fasta_prediction
import streamlit as st


def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    st.sidebar.title("ZeroTag")

    person = st.sidebar.radio("Select person:", ["Person 1", "Person 2"])
    config_path = "config/config_person1.yaml" if person == "Person 1" else "config/config_person2.yaml"
    config = load_config(config_path)
    example_tagging_app(config)
    # else:
    #     app_fasta_prediction(config)

if __name__ == "__main__":
    main()