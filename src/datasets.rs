use csv::Reader;
use ndarray::Array2;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct IrisRecord {
    sepal_length: f64,
    sepal_width: f64,
    petal_length: f64,
    petal_width: f64,
    species: String,
}

pub fn load_iris() -> Result<(Array2<f64>, Vec<String>), Box<dyn std::error::Error>> {
    let mut rdr = Reader::from_path("data/iris.csv")?;
    let mut records = Vec::new();
    let mut labels = Vec::new();

    for result in rdr.deserialize() {
        let record: IrisRecord = result?;
        records.push(vec![
            record.sepal_length,
            record.sepal_width,
            record.petal_length,
            record.petal_width,
        ]);
        labels.push(record.species);
    }

    let data = Array2::from_shape_vec((records.len(), 4), records.into_iter().flatten().collect())?;
    Ok((data, labels))
}
