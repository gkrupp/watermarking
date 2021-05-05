const fs = require('fs')
const pathlib = require('path')
const sharp = require('sharp')

const pathOriginals  = './originals'
const pathMonochrome = './monochrome'

const originals = fs.readdirSync(pathOriginals)
for (const filename of originals) {
	const name = filename.split('.').slice(0,-1).join('.')
	const inputpath = pathlib.join(pathOriginals, filename)
	
	// monochrome
	for (const size of [512, 256, 128, 64, 32]) {
		const outputpath = pathlib.join(pathMonochrome, `${name}_${size}.png`)
		sharp(inputpath)
			.rotate()
			.greyscale()
			.resize({
				width: size,
				height: size,
				fit: 'cover',
				position: 'centre',
				kernel: 'lanczos3'
			})
			.toFile(outputpath)
	}
}
