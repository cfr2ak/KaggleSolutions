using CSV


@info("Loading mnist data from csv file")
train_df = CSV.file("Data/train.csv") |> DataFrame!
train_label = train_df["label"]





