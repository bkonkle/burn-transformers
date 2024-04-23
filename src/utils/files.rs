use tokio::{
    fs::File,
    io::{self, AsyncBufReadExt, Lines},
};

/// Read a file from the given path into a list of strings
pub async fn read_file(path: &str) -> io::Result<Vec<String>> {
    let mut r = file_reader(path).await?;
    let mut lines = Vec::new();

    while let Some(line) = r.next_line().await? {
        lines.push(line);
    }

    Ok(lines)
}

async fn file_reader(path: &str) -> io::Result<Lines<io::BufReader<File>>> {
    let f = File::open(path).await?;

    Ok(io::BufReader::new(f).lines())
}
