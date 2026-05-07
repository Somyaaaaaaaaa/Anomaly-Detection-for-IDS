import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(train_path, test_path):

    train_df = pd.read_csv(train_path, header=None)

    test_df = pd.read_csv(test_path, header=None)

    return train_df, test_df


def preprocess_data(train_df, test_df):
    column_names = [
        "duration",
        "protocol_type",
        "service",
        "flag",
        "src_bytes",
        "dst_bytes",
        "land",
        "wrong_fragment",
        "urgent",
        "hot",
        "num_failed_logins",
        "logged_in",
        "num_compromised",
        "root_shell",
        "su_attempted",
        "num_root",
        "num_file_creations",
        "num_shells",
        "num_access_files",
        "num_outbound_cmds",
        "is_host_login",
        "is_guest_login",
        "count",
        "srv_count",
        "serror_rate",
        "srv_serror_rate",
        "rerror_rate",
        "srv_rerror_rate",
        "same_srv_rate",
        "diff_srv_rate",
        "srv_diff_host_rate",
        "dst_host_count",
        "dst_host_srv_count",
        "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate",
        "dst_host_srv_serror_rate",
        "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate",
        "label",
        "difficulty"
    ]

    train_df.columns = column_names
    test_df.columns = column_names

    return train_df, test_df

def scale_data(train_df, test_df):
    train_df['binary_label'] = train_df["label"].apply(lambda x: 0 if x == "normal" else 1)
    test_df['binary_label'] = test_df["label"].apply(lambda x: 0 if x == "normal" else 1)

    categorical_cols = ["protocol_type", "service", "flag"]

    train_df = pd.get_dummies(train_df, columns=categorical_cols)
    test_df = pd.get_dummies(test_df, columns=categorical_cols)

    train_df, test_df = train_df.align(test_df, join='left', axis=1, fill_value=0)

    X_train = train_df.drop(columns=["label", "binary_label", "difficulty"])
    X_test = test_df.drop(columns=["label", "binary_label", "difficulty"])

    y_test = test_df["binary_label"]

    scaler= MinMaxScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_test, train_df, test_df   

def extract_normal_data(X_train, train_df):

    normal_indices = (
        train_df['binary_label'] == 0
    )

    X_training_set = X_train[normal_indices]

    return X_training_set