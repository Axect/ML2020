#/bin/sh

RUSTFLAGS='-C target-cpu=native' cargo run --release --bin 05_06_mahal
python nc_plot.py
