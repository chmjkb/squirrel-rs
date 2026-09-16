use std::io;

pub fn read_bool<R: io::Read>(r: &mut R) -> io::Result<bool> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    match u8::from_le_bytes(buf) {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid bool data!",
        )),
    }
}

pub fn read_i8<R: io::Read>(r: &mut R) -> io::Result<i8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(i8::from_le_bytes(buf))
}

pub fn read_i16<R: io::Read>(r: &mut R) -> io::Result<i16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(i16::from_le_bytes(buf))
}

pub fn read_i32<R: io::Read>(r: &mut R) -> io::Result<i32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

pub fn read_i64<R: io::Read>(r: &mut R) -> io::Result<i64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(i64::from_le_bytes(buf))
}

pub fn read_u8<R: io::Read>(r: &mut R) -> io::Result<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(u8::from_le_bytes(buf))
}

pub fn read_u16<R: io::Read>(r: &mut R) -> io::Result<u16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

pub fn read_u32<R: io::Read>(r: &mut R) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

pub fn read_u64<R: io::Read>(r: &mut R) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

pub fn read_f32<R: io::Read>(r: &mut R) -> io::Result<f32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

pub fn read_f64<R: io::Read>(r: &mut R) -> io::Result<f64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

pub fn read_str<R: io::Read>(r: &mut R) -> io::Result<String> {
    let raw_len = read_u64(r)?;
    let length =
        usize::try_from(raw_len).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let mut buf = vec![0u8; length];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}
