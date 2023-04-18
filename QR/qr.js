// Inspired from https://kazuhikoarase.github.io/qrcode-generator/js/demo/
//
// Licensed under the MIT license:
//    http://www.opensource.org/licenses/mit-license.php
//
// The word 'QR Code' is registered trademark of
// DENSO WAVE INCORPORATED
//   http://www.denso-wave.com/qrcode/faqpatent-e.html
//

////////////////////////////////////////////////////////////////////////////////
// Drag'n'Drop

function PreventDefaults (e) {
  e.preventDefault();
  e.stopPropagation();
}
function HandleDrop(e) {
  HandleFile(e.dataTransfer.files[0]);
}
function HandleFile(file) {
  const reader = new FileReader();
  reader.readAsDataURL(file);
  reader.onloadend = function() { SetImage(reader.result); }
}
function SetupDragAndDrop() {
  const dropArea = document.getElementById('drop-area');
  function highlight(e) { dropArea.classList.add('highlight'); }
  function unhighlight(e) { dropArea.classList.remove('highlight'); }

  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(
    name => { dropArea.addEventListener(name, PreventDefaults, false); }
  );
  ['dragenter', 'dragover'].forEach(
    name => { dropArea.addEventListener(name, highlight, false); }
  );
  ['dragleave', 'drop'].forEach(
    name => { dropArea.addEventListener(name, unhighlight, false); }
  );
  dropArea.addEventListener('drop', HandleDrop, false)
}

////////////////////////////////////////////////////////////////////////////////

var params = {  // Global parameters
  pixelSize: 2,
  dithering: 'diffusion',
  num_levels: 2,
  luma_adjust: 128,
  image: null,
  text: "https://skal65535.github.io/QR",
  QRsize: 6,   // in [1, 10], 0=Auto
  error_level: 'H',
  background: 'image',
  print() {
    console.log("text = [" + this.text + "]\n" +
                "lvl  = [" + this.errorLevel + "]\n" +
                "QRsize = [" + this.QRsize + "]\n" +
                "Background = [" + this.background + "]\n" +
                "dithering = [" + this.dithering + "]\n" +
                "num_levels = [" + this.num_levels + "]\n" +
                "luma_adjust = [" + this.luma_adjust + "]\n" +
                "pixelSize = [" + this.pixelSize + "]\n");
  },
};

const qrcode = function() {
  //---------------------------------------------------------------------
  // qrcode
  //---------------------------------------------------------------------

  const qrcode = function(typeNumber, errorCorrectLevel) {
    const PAD0 = 0xEC;
    const PAD1 = 0x11;
    const _typeNumber = typeNumber;
    const _errorCorrectLevel = QRErrorCorrectLevel[errorCorrectLevel];
    let _modules = null;
    let _moduleCount = 0;
    let _dataCache = null;
    let _dataList = new Array();
    let _this = {};

    function MakeData(test, pattern_id, onlyControl) {
      _moduleCount = _typeNumber * 4 + 17;
      _modules = function() {
        let modules = new Array(_moduleCount);
        for (let y = 0; y < _moduleCount; ++y) {
          modules[y] = new Array(_moduleCount);
          for (let x = 0; x < _moduleCount; ++x) modules[y][x] = null;
        }
        return modules;
      } ();

      SetupPositionProbePattern(3, 3);
      SetupPositionProbePattern(_moduleCount - 4, 3);
      SetupPositionProbePattern(3, _moduleCount - 4);
      SetupPositionAdjustPattern();
      SetupTimingPattern();
      SetupTypeInfo(test, pattern_id);
      SetupTypeNumber(_typeNumber, test);

      if (!onlyControl) {
        if (_dataCache == null) {
          _dataCache = CreateData(_typeNumber, _errorCorrectLevel, _dataList);
        }
        MapData(_dataCache, pattern_id);
      }
    }

    function GetBestPattern() {
      let best_score = 0;
      let best_pattern_id = 0;
      for (let pattern_id = 0; pattern_id <= 7; ++pattern_id) {
        MakeData(true, pattern_id, false);
        const score = QRUtil.getLostPoint(_this);
        if (pattern_id == 0 || score < best_score) {
          best_score = score;
          best_pattern_id = pattern_id;
        }
      }
      return best_pattern_id;
    }

    function SetupPositionProbePattern(Y, X) {
      for (let y = -4; y <= 4; ++y) {
        if (Y + y < 0 || Y + y >= _moduleCount) continue;
        for (let x = -4; x <= 4; ++x) {
          if (X + x < 0 || X + x >= _moduleCount) continue;
          const v = Math.max(Math.abs(x), Math.abs(y));
          _modules[Y + y][X + x] = (v != 2 && v != 4);
        }
      }
    }
    function SetupPositionAdjustPattern() {
      const pos = QRUtil.getPatternPosition(_typeNumber);
      for (let j = 0; j < pos.length; ++j) {
        for (let i = 0; i < pos.length; ++i) {
          const X = pos[i], Y = pos[j];
          if (_modules[Y][X] != null) continue;
          for (let y = -2; y <= 2; ++y) {
            for (let x = -2; x <= 2; ++x) {
              const mod = (Math.max(Math.abs(x), Math.abs(y)) != 1);
              _modules[Y + y][X + x] = mod;
            }
          }
        }
      }
    }
    function SetupTimingPattern() {
      for (let i = 8; i < _moduleCount - 8; ++i) {
        if (_modules[i][6] == null) _modules[i][6] = (i % 2 == 0);
        if (_modules[6][i] == null) _modules[6][i] = (i % 2 == 0);
      }
    }
    function SetupTypeNumber(typenumber, test) {
      if (typenumber < 7) return;
      const bits = QRUtil.getBCHTypeNumber(typenumber);
      for (let i = 0; i < 18; ++i) {
        const Y = Math.floor(i / 3), X = (i % 3) + _moduleCount - 8 - 3;
        const mod = (!test && ((bits >> i) & 1) == 1);
        _modules[Y][X] = _modules[X][Y] = mod;
      }
    }
    function SetupTypeInfo(test, pattern_id) {
      const data = (_errorCorrectLevel << 3) | pattern_id;
      const bits = QRUtil.getBCHTypeInfo(data);
      for (let i = 0; i < 15; i += 1) {
        const mod = (!test && ( (bits >> i) & 1) == 1);
        // vertical
        if (i < 6) {
          _modules[i][8] = mod;
        } else if (i < 8) {
          _modules[i + 1][8] = mod;
        } else {
          _modules[_moduleCount - 15 + i][8] = mod;
        }
        // horizontal
        if (i < 8) {
          _modules[8][_moduleCount - i - 1] = mod;
        } else if (i < 9) {
          _modules[8][15 - i - 1 + 1] = mod;
        } else {
          _modules[8][15 - i - 1] = mod;
        }
      }
      // fixed module
      _modules[_moduleCount - 8][8] = (!test);
    }
    function MapData(data, pattern_id) {
      let inc = -1;
      let row = _moduleCount - 1;
      let bitIndex = 7;
      let byteIndex = 0;
      let maskFunc = QRUtil.getMaskFunction(pattern_id);
      for (let col = _moduleCount - 1; col > 0; col -= 2) {
        if (col == 6) col -= 1;
        while (true) {
          for (let c = 0; c < 2; c += 1) {
            if (_modules[row][col - c] == null) {
              let dark = false;
              if (byteIndex < data.length) {
                dark = ( ( (data[byteIndex] >>> bitIndex) & 1) == 1);
              }
              const mask = maskFunc(row, col - c);
              if (mask) dark = !dark;
              _modules[row][col - c] = dark;
              bitIndex -= 1;
              if (bitIndex == -1) {
                byteIndex += 1;
                bitIndex = 7;
              }
            }
          }
          row += inc;
          if (row < 0 || _moduleCount <= row) {
            row -= inc;
            inc = -inc;
            break;
          }
        }
      }
    }
    function createBytes(buffer, rsBlocks) {
      let offset = 0;
      let maxDcCount = 0, maxEcCount = 0;
      let dcdata = new Array(rsBlocks.length);
      let ecdata = new Array(rsBlocks.length);
      for (let r = 0; r < rsBlocks.length; r += 1) {
        let dcCount = rsBlocks[r].dataCount;
        let ecCount = rsBlocks[r].totalCount - dcCount;
        maxDcCount = Math.max(maxDcCount, dcCount);
        maxEcCount = Math.max(maxEcCount, ecCount);
        dcdata[r] = new Array(dcCount);
        for (let i = 0; i < dcdata[r].length; i += 1) {
          dcdata[r][i] = 0xff & buffer.getBuffer()[i + offset];
        }
        offset += dcCount;
        let rsPoly = QRUtil.getErrorCorrectPolynomial(ecCount);
        let rawPoly = qrPolynomial(dcdata[r], rsPoly.getLength() - 1);
        let modPoly = rawPoly.mod(rsPoly);
        ecdata[r] = new Array(rsPoly.getLength() - 1);
        for (let i = 0; i < ecdata[r].length; i += 1) {
          const modIndex = i + modPoly.getLength() - ecdata[r].length;
          ecdata[r][i] = (modIndex >= 0)? modPoly.getAt(modIndex) : 0;
        }
      }
      let totalCodeCount = 0;
      for (let i = 0; i < rsBlocks.length; i += 1) {
        totalCodeCount += rsBlocks[i].totalCount;
      }
      let data = new Array(totalCodeCount);
      let index = 0;
      for (let i = 0; i < maxDcCount; i += 1) {
        for (let r = 0; r < rsBlocks.length; r += 1) {
          if (i < dcdata[r].length) {
            data[index++] = dcdata[r][i];
          }
        }
      }
      for (let i = 0; i < maxEcCount; i += 1) {
        for (let r = 0; r < rsBlocks.length; r += 1) {
          if (i < ecdata[r].length) {
            data[index++] = ecdata[r][i];
          }
        }
      }
      return data;
    }

    function CreateData(typeNumber, errorCorrectLevel, dataList) {
      const rsBlocks = QRRSBlock.getRSBlocks(typeNumber, errorCorrectLevel);
      let buffer = qrBitBuffer();
      for (let i = 0; i < dataList.length; i += 1) {
        let data = dataList[i];
        buffer.put(data.getMode(), 4);
        buffer.put(data.getLength(), QRUtil.getLengthInBits(data.getMode(), typeNumber) );
        data.write(buffer);
      }
      // calc num max data.
      let totalDataCount = 0;
      for (let i = 0; i < rsBlocks.length; i += 1) {
        totalDataCount += rsBlocks[i].dataCount;
      }
      if (buffer.getLengthInBits() > totalDataCount * 8) {
        throw new Error('code length overflow. (' + buffer.getLengthInBits() +
                        '>' + totalDataCount * 8 + ')');
      }
      // end code
      if (buffer.getLengthInBits() + 4 <= totalDataCount * 8) buffer.put(0, 4);
      // padding
      while (buffer.getLengthInBits() % 8 != 0) buffer.putBit(false);
      // padding
      while (true) {
        if (buffer.getLengthInBits() >= totalDataCount * 8) break;
        buffer.put(PAD0, 8);
        if (buffer.getLengthInBits() >= totalDataCount * 8) break;
        buffer.put(PAD1, 8);
      }
      return createBytes(buffer, rsBlocks);
    }
    _this.addData = function(data) {
      const newData = qr8BitByte(data);
      _dataList.push(newData);
      _dataCache = null;
    };
    _this.isDark = function(Y, X) {
      if (Y < 0 || Y >= _moduleCount || X < 0 || X >= _moduleCount) {
        throw new Error(X + ',' + Y);
      }
      return _modules[Y][X];
    };
    _this.make = function(onlyControl) {
      const best_pattern_id = GetBestPattern()
      MakeData(false, best_pattern_id, onlyControl);
    };
    _this.returnByteArray = function() { return _modules; }

    return _this;
  };

//---------------------------------------------------------------------
// qrcode.stringToBytes
//---------------------------------------------------------------------

function StringToBytes(s) {
  let bytes = new Array();
  for (let i = 0; i < s.length; i += 1) {
    const c = s.charCodeAt(i);
    bytes.push(c & 0xff);
  }
  return bytes;
}

  // QRMode
  const QRMode = {
    MODE_NUMBER :     1 << 0,
    MODE_ALPHA_NUM :  1 << 1,
    MODE_8BIT_BYTE :  1 << 2,
    MODE_KANJI :      1 << 3
  };
  qrcode.QRMode = QRMode;

  // QRErrorCorrectLevel
  const QRErrorCorrectLevel = { L : 1, M : 0, Q : 3, H : 2 };
  qrcode.QRErrorCorrectLevel = QRErrorCorrectLevel;

  // QRMaskPattern
  const QRMaskPattern = {
    PATTERN000 : 0,
    PATTERN001 : 1,
    PATTERN010 : 2,
    PATTERN011 : 3,
    PATTERN100 : 4,
    PATTERN101 : 5,
    PATTERN110 : 6,
    PATTERN111 : 7
  };
  qrcode.QRMaskPattern = QRMaskPattern;

  // QRUtil
  const QRUtil = function() {
    const PATTERN_POSITION_TABLE = [
			[],
			[6, 18],
			[6, 22],
			[6, 26],
			[6, 30],
			[6, 34],
			[6, 22, 38],
			[6, 24, 42],
			[6, 26, 46],
			[6, 28, 50],
			[6, 30, 54],
			[6, 32, 58],
			[6, 34, 62],
			[6, 26, 46, 66],
			[6, 26, 48, 70],
			[6, 26, 50, 74],
			[6, 30, 54, 78],
			[6, 30, 56, 82],
			[6, 30, 58, 86],
			[6, 34, 62, 90],
			[6, 28, 50, 72, 94],
			[6, 26, 50, 74, 98],
			[6, 30, 54, 78, 102],
			[6, 28, 54, 80, 106],
			[6, 32, 58, 84, 110],
			[6, 30, 58, 86, 114],
			[6, 34, 62, 90, 118],
			[6, 26, 50, 74, 98, 122],
			[6, 30, 54, 78, 102, 126],
			[6, 26, 52, 78, 104, 130],
			[6, 30, 56, 82, 108, 134],
			[6, 34, 60, 86, 112, 138],
			[6, 30, 58, 86, 114, 142],
			[6, 34, 62, 90, 118, 146],
			[6, 30, 54, 78, 102, 126, 150],
			[6, 24, 50, 76, 102, 128, 154],
			[6, 28, 54, 80, 106, 132, 158],
			[6, 32, 58, 84, 110, 136, 162],
			[6, 26, 54, 82, 110, 138, 166],
			[6, 30, 58, 86, 114, 142, 170]
    ];
    const G15 = (1 << 10) | (1 << 8) | (1 << 5) | (1 << 4) | (1 << 2) | (1 << 1) | (1 << 0);
    const G18 = (1 << 12) | (1 << 11) | (1 << 10) | (1 << 9) | (1 << 8) | (1 << 5) | (1 << 2) | (1 << 0);
    const G15_MASK = (1 << 14) | (1 << 12) | (1 << 10) | (1 << 4) | (1 << 1);
    let _this = {};
    const getBCHDigit = function(data) {
      let digit = 0;
      while (data != 0) {
        digit += 1;
        data >>>= 1;
      }
      return digit;
    };
    _this.getBCHTypeInfo = function(data) {
      let d = data << 10;
      while (getBCHDigit(d) - getBCHDigit(G15) >= 0) {
        d ^= (G15 << (getBCHDigit(d) - getBCHDigit(G15) ) );
      }
      return ( (data << 10) | d) ^ G15_MASK;
    };
    _this.getBCHTypeNumber = function(data) {
      let d = data << 12;
      while (getBCHDigit(d) - getBCHDigit(G18) >= 0) {
        d ^= (G18 << (getBCHDigit(d) - getBCHDigit(G18) ) );
      }
      return (data << 12) | d;
    };
    _this.getPatternPosition = function(typeNumber) {
      return PATTERN_POSITION_TABLE[typeNumber - 1];
    };
    _this.getMaskFunction = function(pattern_id) {
      switch (pattern_id) {
        case QRMaskPattern.PATTERN000 : return function(i, j) { return (i + j) % 2 == 0; };
        case QRMaskPattern.PATTERN001 : return function(i, j) { return i % 2 == 0; };
        case QRMaskPattern.PATTERN010 : return function(i, j) { return j % 3 == 0; };
        case QRMaskPattern.PATTERN011 : return function(i, j) { return (i + j) % 3 == 0; };
        case QRMaskPattern.PATTERN100 : return function(i, j) { return (Math.floor(i / 2) + Math.floor(j / 3) ) % 2 == 0; };
        case QRMaskPattern.PATTERN101 : return function(i, j) { return (i * j) % 2 + (i * j) % 3 == 0; };
        case QRMaskPattern.PATTERN110 : return function(i, j) { return ( (i * j) % 2 + (i * j) % 3) % 2 == 0; };
        case QRMaskPattern.PATTERN111 : return function(i, j) { return ( (i * j) % 3 + (i + j) % 2) % 2 == 0; };
        default : throw new Error('bad maskPattern:' + pattern_id);
      }
    };
    _this.getErrorCorrectPolynomial = function(errorCorrectLength) {
      let a = qrPolynomial([1], 0);
      for (let i = 0; i < errorCorrectLength; ++i) {
        a = a.multiply(qrPolynomial([1, QRMath.gexp(i)], 0));
      }
      return a;
    };
    _this.getLengthInBits = function(mode, type) {
      if (1 <= type && type < 10) {

				// 1 - 9

				switch(mode) {
				case QRMode.MODE_NUMBER 	: return 10;
				case QRMode.MODE_ALPHA_NUM 	: return 9;
				case QRMode.MODE_8BIT_BYTE	: return 8;
				case QRMode.MODE_KANJI		: return 8;
				default :
					throw new Error('mode:' + mode);
				}

			} else if (type < 27) {

				// 10 - 26

				switch(mode) {
				case QRMode.MODE_NUMBER 	: return 12;
				case QRMode.MODE_ALPHA_NUM 	: return 11;
				case QRMode.MODE_8BIT_BYTE	: return 16;
				case QRMode.MODE_KANJI		: return 10;
				default :
					throw new Error('mode:' + mode);
				}

			} else if (type < 41) {

				// 27 - 40

				switch(mode) {
				case QRMode.MODE_NUMBER 	: return 14;
				case QRMode.MODE_ALPHA_NUM	: return 13;
				case QRMode.MODE_8BIT_BYTE	: return 16;
				case QRMode.MODE_KANJI		: return 12;
				default :
					throw new Error('mode:' + mode);
				}

			} else {
				throw new Error('type:' + type);
			}
		};

		_this.getLostPoint = function(qrcode) {

			let moduleCount = qrcode._moduleCount;

			let lostPoint = 0;

			// LEVEL1

			for (let row = 0; row < moduleCount; row += 1) {
				for (let col = 0; col < moduleCount; col += 1) {

					let sameCount = 0;
					let dark = qrcode.isDark(row, col);

					for (let r = -1; r <= 1; r += 1) {

						if (row + r < 0 || moduleCount <= row + r) {
							continue;
						}

						for (let c = -1; c <= 1; c += 1) {

							if (col + c < 0 || moduleCount <= col + c) {
								continue;
							}

							if (r == 0 && c == 0) {
								continue;
							}

							if (dark == qrcode.isDark(row + r, col + c) ) {
								sameCount += 1;
							}
						}
					}

					if (sameCount > 5) {
						lostPoint += (3 + sameCount - 5);
					}
				}
			};

			// LEVEL2

			for (let row = 0; row < moduleCount - 1; row += 1) {
				for (let col = 0; col < moduleCount - 1; col += 1) {
					let count = 0;
					if (qrcode.isDark(row, col) ) count += 1;
					if (qrcode.isDark(row + 1, col) ) count += 1;
					if (qrcode.isDark(row, col + 1) ) count += 1;
					if (qrcode.isDark(row + 1, col + 1) ) count += 1;
					if (count == 0 || count == 4) {
						lostPoint += 3;
					}
				}
			}

			// LEVEL3

			for (let row = 0; row < moduleCount; row += 1) {
				for (let col = 0; col < moduleCount - 6; col += 1) {
					if (qrcode.isDark(row, col)
							&& !qrcode.isDark(row, col + 1)
							&&  qrcode.isDark(row, col + 2)
							&&  qrcode.isDark(row, col + 3)
							&&  qrcode.isDark(row, col + 4)
							&& !qrcode.isDark(row, col + 5)
							&&  qrcode.isDark(row, col + 6) ) {
						lostPoint += 40;
					}
				}
			}

			for (let col = 0; col < moduleCount; col += 1) {
				for (let row = 0; row < moduleCount - 6; row += 1) {
					if (qrcode.isDark(row, col)
							&& !qrcode.isDark(row + 1, col)
							&&  qrcode.isDark(row + 2, col)
							&&  qrcode.isDark(row + 3, col)
							&&  qrcode.isDark(row + 4, col)
							&& !qrcode.isDark(row + 5, col)
							&&  qrcode.isDark(row + 6, col) ) {
						lostPoint += 40;
					}
				}
			}

			// LEVEL4

			let darkCount = 0;

			for (let col = 0; col < moduleCount; col += 1) {
				for (let row = 0; row < moduleCount; row += 1) {
					if (qrcode.isDark(row, col) ) {
						darkCount += 1;
					}
				}
			}

			let ratio = Math.abs(100 * darkCount / moduleCount / moduleCount - 50) / 5;
			lostPoint += ratio * 10;

			return lostPoint;
		};

		return _this;
	}();
	qrcode.QRUtil = QRUtil;

  // QRMath
  const QRMath = function() {
    let EXP_TABLE = new Array(256);
    let LOG_TABLE = new Array(256);
    // initialize tables
    for (let i = 0; i < 8; ++i) EXP_TABLE[i] = 1 << i;
    for (let i = 8; i < 256; ++i) {
      EXP_TABLE[i] = EXP_TABLE[i - 4]
                   ^ EXP_TABLE[i - 5]
                   ^ EXP_TABLE[i - 6]
                   ^ EXP_TABLE[i - 8];
    }
    for (let i = 0; i < 255; ++i) LOG_TABLE[EXP_TABLE[i]] = i;
    let _this = {};
    _this.glog = function(n) {
      if (n < 1) throw new Error('glog(' + n + ')');
      return LOG_TABLE[n];
    };
    _this.gexp = function(n) {
      while (n < 0) n += 255;
      while (n >= 256) n -= 255;
      return EXP_TABLE[n];
    };

    return _this;
  }();

  // qrPolynomial
  function qrPolynomial(num, shift) {
    if (typeof num.length == 'undefined') {
      throw new Error(num.length + '/' + shift);
    }
    const _num = function() {
      let offset = 0;
      while (offset < num.length && num[offset] == 0) offset += 1;
      const _num = new Array(num.length - offset + shift);
      for (let i = 0; i < num.length - offset; i += 1) {
        _num[i] = num[i + offset];
      }
      return _num;
    }();
    let _this = {};
    _this.getAt = function(index) { return _num[index]; };
    _this.getLength = function() { return _num.length; };
    _this.multiply = function(e) {
      let num = new Array(_this.getLength() + e.getLength() - 1);
      for (let i = 0; i < _this.getLength(); ++i) {
        for (let j = 0; j < e.getLength(); ++j) {
          num[i + j] ^= QRMath.gexp(QRMath.glog(_this.getAt(i) ) + QRMath.glog(e.getAt(j) ) );
        }
      }
      return qrPolynomial(num, 0);
    };
    _this.mod = function(e) {
      if (_this.getLength() - e.getLength() < 0) return _this;
      let ratio = QRMath.glog(_this.getAt(0) ) - QRMath.glog(e.getAt(0) );
      let num = new Array(_this.getLength() );
      for (let i = 0; i < _this.getLength(); ++i) num[i] = _this.getAt(i);
      for (let i = 0; i < e.getLength(); ++i) {
        num[i] ^= QRMath.gexp(QRMath.glog(e.getAt(i) ) + ratio);
      }
      // recursive call
      return qrPolynomial(num, 0).mod(e);
    };

    return _this;
  };

  // QRRSBlock
  const QRRSBlock = function() {
    let _this = {};
    let RS_BLOCK_TABLE = [   // L, M, Q, H
			// 1
			[1, 26, 19],
			[1, 26, 16],
			[1, 26, 13],
			[1, 26, 9],

			// 2
			[1, 44, 34],
			[1, 44, 28],
			[1, 44, 22],
			[1, 44, 16],

			// 3
			[1, 70, 55],
			[1, 70, 44],
			[2, 35, 17],
			[2, 35, 13],

			// 4
			[1, 100, 80],
			[2, 50, 32],
			[2, 50, 24],
			[4, 25, 9],

			// 5
			[1, 134, 108],
			[2, 67, 43],
			[2, 33, 15, 2, 34, 16],
			[2, 33, 11, 2, 34, 12],

			// 6
			[2, 86, 68],
			[4, 43, 27],
			[4, 43, 19],
			[4, 43, 15],

			// 7
			[2, 98, 78],
			[4, 49, 31],
			[2, 32, 14, 4, 33, 15],
			[4, 39, 13, 1, 40, 14],

			// 8
			[2, 121, 97],
			[2, 60, 38, 2, 61, 39],
			[4, 40, 18, 2, 41, 19],
			[4, 40, 14, 2, 41, 15],

			// 9
			[2, 146, 116],
			[3, 58, 36, 2, 59, 37],
			[4, 36, 16, 4, 37, 17],
			[4, 36, 12, 4, 37, 13],

			// 10
			[2, 86, 68, 2, 87, 69],
			[4, 69, 43, 1, 70, 44],
			[6, 43, 19, 2, 44, 20],
			[6, 43, 15, 2, 44, 16]
		];
    _this.RS_BLOCK_TABLE = RS_BLOCK_TABLE;
    const qrRSBlock = function(totalCount, dataCount) {
      let _this = {};
      _this.totalCount = totalCount;
      _this.dataCount = dataCount;
      return _this;
    };
    const getRsBlockTable = function(typeNumber, errorCorrectLevel) {
      switch(errorCorrectLevel) {
        case QRErrorCorrectLevel.L : return RS_BLOCK_TABLE[(typeNumber - 1) * 4 + 0];
        case QRErrorCorrectLevel.M : return RS_BLOCK_TABLE[(typeNumber - 1) * 4 + 1];
        case QRErrorCorrectLevel.Q : return RS_BLOCK_TABLE[(typeNumber - 1) * 4 + 2];
        case QRErrorCorrectLevel.H : return RS_BLOCK_TABLE[(typeNumber - 1) * 4 + 3];
        default : return undefined;
      }
    };
    _this.getRSBlocks = function(typeNumber, errorCorrectLevel) {
      const rsBlock = getRsBlockTable(typeNumber, errorCorrectLevel);
      if (typeof rsBlock == 'undefined') {
        throw new Error('bad rs block @ typeNumber:' + typeNumber +
                        '/errorCorrectLevel:' + errorCorrectLevel);
      }
      const length = rsBlock.length / 3;
      let list = new Array();
      for (let i = 0; i < length; i += 1) {
        let count = rsBlock[i * 3 + 0];
        let totalCount = rsBlock[i * 3 + 1];
        let dataCount = rsBlock[i * 3 + 2];
        for (let j = 0; j < count; j += 1) {
          list.push(qrRSBlock(totalCount, dataCount) );
        }
      }
      return list;
    };
    return _this;
  }();
  qrcode.QRRSBlock = QRRSBlock;

  // qrBitBuffer
  const qrBitBuffer = function() {
    let _buffer = new Array();
    let _length = 0;
    let _this = {};

    _this.getBuffer = function() { return _buffer; };
    _this.getAt = function(index) {
      const bufIndex = Math.floor(index / 8);
      return ( (_buffer[bufIndex] >>> (7 - index % 8) ) & 1) == 1;
    };
    _this.put = function(num, length) {
      for (let i = 0; i < length; i += 1) {
        _this.putBit( ( (num >>> (length - i - 1) ) & 1) == 1);
      }
    };
    _this.getLengthInBits = function() { return _length; };
    _this.putBit = function(bit) {
      const bufIndex = Math.floor(_length / 8);
      if (_buffer.length <= bufIndex) {
        _buffer.push(0);
      }
      if (bit) {
        _buffer[bufIndex] |= (0x80 >>> (_length % 8) );
      }
      _length += 1;
    };
    return _this;
  };

  // qr8BitByte
  const qr8BitByte = function(data) {
    const _mode = QRMode.MODE_8BIT_BYTE;
    const _bytes = StringToBytes(data);
    let _this = {};

    _this.getMode = function() { return _mode; };
    _this.getLength = function(buffer) { return _bytes.length; };
    _this.write = function(buffer) {
      for (let i = 0; i < _bytes.length; i += 1) buffer.put(_bytes[i], 8);
    };
    return _this;
  };

  //---------------------------------------------------------------------
  // returns qrcode function.
  return qrcode;
}();

function HalftoneQR(QRBytes, controlBytes, image) {
  const pixelSize = params.pixelSize;
  const blockSize = 3 * pixelSize;
  const dim = QRBytes.length * blockSize;

  ['#imageColour', '#imageThreshold', '#imagePixel', '#output'].forEach(name => {
    const obj = document.querySelector(name);
    obj.width = dim;
    obj.height = dim;
  });

  const canvas = document.querySelector('#output');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, dim, dim)
  if (params.image != null) {
    Render();
    if (params.background === 'image') {
      const canvasThreshold = document.querySelector('#imageThreshold');
      const ctxThreshold = canvasThreshold.getContext('2d');
      ctx.drawImage(canvasThreshold, 0, 0, dim, dim);
    }
  }
  for (let Y = 0; Y < QRBytes.length; ++Y) {
    for (let X = 0; X < QRBytes[Y].length; ++X) {
      if (params.image == null || params.background === 'noise') {
        // Draw random bytes
        for (let y = 0; y < 3; ++y) {
          for (let x = 0; x < 3; ++x) {
            ctx.fillStyle = (Math.random() < 0.5) ? 'white' : 'black';
            ctx.fillRect((3 * Y + y) * pixelSize, (3 * X + x) * pixelSize,
                         pixelSize, pixelSize);
          }
        }
      }
      // Re-draw control bytes
      if (controlBytes[Y][X] !== null) {
        ctx.fillStyle = (controlBytes[Y][X] === true) ? 'black' : 'white';
        ctx.fillRect(Y * blockSize, X * blockSize, blockSize, blockSize);
      } else {
        // Middle Cell
        ctx.fillStyle = QRBytes[Y][X] ? 'black' : 'white';
        ctx.fillRect((3 * Y + 1) * pixelSize, (3 * X + 1) * pixelSize,
                     pixelSize, pixelSize);
      }
    }
  }
  const result = document.querySelector('#output').toDataURL('image/webp', 1);
  document.querySelector('#download').href = result;
}

function DisableSmoothing(ctx) {
  ctx.imageSmoothingEnabled = false;
  ctx.mozImageSmoothingEnabled = false;
  ctx.msImageSmoothingEnabled = false;
  ctx.webkitImageSmoothingEnabled =  false;
}

function Render() {
  const canvasColour = document.querySelector('#imageColour');
  const ctxColour = canvasColour.getContext('2d');
  ctxColour.clearRect(0, 0, canvasColour.width, canvasColour.height);
  ctxColour.drawImage(params.image, 0, 0, canvasColour.width, canvasColour.height);
    
  const canvasPixel = document.querySelector('#imagePixel');
  const ctxPixel = canvasPixel.getContext('2d');

  const canvasTmp = document.createElement('canvas');
  canvasTmp.width = canvasTmp.height = (canvasPixel.width / params.pixelSize);
  const ctxTmp = canvasTmp.getContext('2d');
  DisableSmoothing(ctxPixel);
  DisableSmoothing(ctxTmp);
  ctxTmp.drawImage(canvasColour, 0, 0, canvasTmp.width, canvasTmp.height);
  ctxPixel.drawImage(canvasTmp, 0, 0, canvasPixel.width, canvasPixel.height);

  // DitherImage
  const canvasThreshold = document.querySelector('#imageThreshold');
  const ctxThreshold = canvasThreshold.getContext('2d');
    
  let pixels = ctxPixel.getImageData(0, 0, canvasPixel.width, canvasPixel.height);
  let d = pixels.data;
  let num_levels = params.num_levels;
  let adjust = params.luma_adjust;

  for (let i = 0; i < d.length; i += 4) {
    const r = d[i + 0], g = d[i + 1], b = d[i + 2];
    d[i + 0] = Math.floor(r * 0.2126 + g * 0.7152 + b * 0.0722);
    d[i + 1] = 0;   // reset error
  }
  const stride = 4 * canvasPixel.width;
  for (let y = 0; y < canvasPixel.height; ++y) {
    for (let x = 0; x < canvasPixel.width; ++x) {
      const off0 = y * stride + x * 4;
      let correction = 0;
      if (params.dithering == 'diffusion') {
        const off1 = off0 - stride;
        correction = 8;
        correction += 7 * d[off0 - 1 * 4 + 1];
        correction += 1 * d[off1 - 1 * 4 + 1];
        correction += 5 * d[off1 + 0 * 4 + 1];
        correction += 3 * d[off1 + 1 * 4 + 1];
        correction /= 16;
      } else if (params.dithering == 'random') {
        correction = Math.floor(Math.random() * 255);
      } else if (params.dithering == 'ordered') {
        const dithering_4x4 = [ 0, 8, 2, 10,  12, 4, 14, 6,  3, 11, 1, 9,  15, 7, 13, 5 ];
        const x4 = x % 4, y4 = y % 4;
        correction = dithering_4x4[x4 + 4 * y4] * 4;
      }
      const gray = Math.max(0, Math.min(d[off0 + 0] + correction, 255));
      const q_gray = Math.floor(gray * num_levels / 256);
      const d_gray = Math.round((q_gray * 255 + adjust) / num_levels);
      const error = gray - d_gray;
      d[off0 + 0] = d_gray;
      d[off0 + 1] = error;
    }
  }
  for (let i = 0; i < d.length; i += 4) {
    d[i + 1] = d[i + 2] = d[i + 0];
    d[i + 3] = 255;
  }
  ctxThreshold.putImageData(pixels, 0, 0);
}

function ParseParams(p) {
  p.text = document.querySelector('#text').value;
  p.errorLevel = document.querySelector('#error_level_selector').value;
  p.pixelSize = parseInt(document.querySelector('#pixel_size_selector').value);
  p.dithering = document.querySelector('#dithering_selector').value;
  p.num_levels = document.querySelector('#gray_selector').value;
  p.luma_adjust = parseInt(document.querySelector('#luma_selector').value);
  p.background = document.querySelector('#background').value;

  const sizes = {
    // 10 levels for each quality
    L: [152, 272, 440, 640, 864, 1088, 1248, 1552, 1856, 1240],
    M: [128, 224, 352, 512, 688, 864,  992,  700,  700,  524],
    Q: [104, 176, 272, 384, 286, 608,  508,  376,  608,  434],
    H: [72,  128, 208, 288, 214, 480,  164,  296,  464,  346]
  };
  const userSize = parseInt(document.querySelector('#size_selector').value);

  p.QRsize = 0;
  if (userSize === 0) {    // 'Auto'
    for (let i = 0; i < sizes[p.errorLevel].length; i++) {
      if (p.text.length < sizes[p.errorLevel][i]) {
        p.QRsize = i + 1;
        break;
      }
    }
    if (p.QRsize === 0) {
      if (p.errorLevel === 'H') {
        alert('Too much text.');
      } else {
        alert('Too much text. Try decreasing the error level.');
      }
    }
  } else {
    if (p.text.length < sizes[p.errorLevel][userSize - 1]) {
      p.QRsize = userSize;
    } else {
      alert('Text is too long (length: ' + p.text.length +
            '). Try decreasing the error level (' + p.errorLevel +
            ') or increasing the size (' + userSize + ').');
    }
  }
  if (p.QRsize == 0) return false;
  return true;
}

function GenerateQRCode() {
  let new_params = params;
  if (!ParseParams(new_params)) return;
  params = new_params;
  params.print();

  const qr = qrcode(params.QRsize, params.errorLevel);
  qr.addData(params.text);
  qr.make(false);

  const controls = qrcode(params.QRsize, params.errorLevel);
  controls.addData(params.text);
  controls.make(true);

  HalftoneQR(qr.returnByteArray(), controls.returnByteArray());        
}

function SetImage(url) {
  const new_image = new Image();
  new_image.onload = function() {
    params.image = this;
    GenerateQRCode();
  }
  new_image.src = url;
}
