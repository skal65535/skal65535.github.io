// Standalone QR Code generator
//
// Inspired from https://kazuhikoarase.github.io/qrcode-generator/js/demo/
//
// Licensed under the MIT license:
//   http://www.opensource.org/licenses/mit-license.php
//
// The word 'QR Code' is registered trademark of
// DENSO WAVE INCORPORATED
//   http://www.denso-wave.com/qrcode/faqpatent-e.html
//

const PAD0 = 0xEC;
const PAD1 = 0x11;

// QRRSBlock
const QRRSBlock = function() {
  const RSBlockTable = [   // L, M, Q, H
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
  function GetRsBlockTable(typeNumber, errorCorrectLevel) {
    switch(errorCorrectLevel) {
      case QRErrorCorrectLevel.L : return RSBlockTable[(typeNumber - 1) * 4 + 0];
      case QRErrorCorrectLevel.M : return RSBlockTable[(typeNumber - 1) * 4 + 1];
      case QRErrorCorrectLevel.Q : return RSBlockTable[(typeNumber - 1) * 4 + 2];
      case QRErrorCorrectLevel.H : return RSBlockTable[(typeNumber - 1) * 4 + 3];
      default : return undefined;
    }
  };

  let _this = {};
  _this.GetRSBlocks = function(typeNumber, errorCorrectLevel) {
    const rsBlock = GetRsBlockTable(typeNumber, errorCorrectLevel);
    if (typeof rsBlock == 'undefined') {
      throw new Error('bad rs block @ typeNumber:' + typeNumber +
                      '/errorCorrectLevel:' + errorCorrectLevel);
    }
    let list = new Array();
    for (let i = 0; i < rsBlock.length; i += 3) {
      const num_blocks = rsBlock[i + 0];
      const qrRSBlock = { totalCount: rsBlock[i + 1], dataCount: rsBlock[i + 2] };
      for (let j = 0; j < num_blocks; ++j) list.push(qrRSBlock);
    }
    return list;
  };
  return _this;
}();

//---------------------------------------------------------------------

function CreateBytes(buffer, rsBlocks) {
  let offset = 0;
  let maxDcCount = 0, maxEcCount = 0;
  let dcdata = new Array(rsBlocks.length);
  let ecdata = new Array(rsBlocks.length);
  for (let r = 0; r < rsBlocks.length; ++r) {
    const dcCount = rsBlocks[r].dataCount;
    const ecCount = rsBlocks[r].totalCount - dcCount;
    maxDcCount = Math.max(maxDcCount, dcCount);
    maxEcCount = Math.max(maxEcCount, ecCount);
    dcdata[r] = buffer.getBuffer().slice(offset, offset + dcCount);
    offset += dcCount;
    let rsPoly = GetErrorCorrectPolynomial(ecCount);
    let rawPoly = qrPolynomial(dcdata[r], rsPoly.getLength() - 1);
    let modPoly = rawPoly.mod(rsPoly);
    ecdata[r] = new Array(rsPoly.getLength() - 1);
    for (let i = 0; i < ecdata[r].length; ++i) {
      const modIndex = i + modPoly.getLength() - ecdata[r].length;
      ecdata[r][i] = (modIndex >= 0)? modPoly.getAt(modIndex) : 0;
    }
  }
  let totalCodeCount = 0;
  for (const b of rsBlocks) totalCodeCount += b.totalCount;

  let data = new Array(totalCodeCount);
  let index = 0;
  for (let i = 0; i < maxDcCount; ++i) {
    for (let r = 0; r < rsBlocks.length; ++r) {
      if (i < dcdata[r].length) data[index++] = dcdata[r][i];
    }
  }
  for (let i = 0; i < maxEcCount; ++i) {
    for (let r = 0; r < rsBlocks.length; ++r) {
      if (i < ecdata[r].length) data[index++] = ecdata[r][i];
    }
  }
  return data;
}

//---------------------------------------------------------------------
// BitBuffer

function GetLengthInBits(mode, type) {
    if (1 <= type && type < 10) {   // 1 - 9
      switch(mode) {
        case QRMode.NUMBER         : return 10;
        case QRMode.ALPHA_NUM      : return 9;
        case QRMode.BYTE_8BIT      : return 8;
        case QRMode.KANJI          : return 8;
      }
    } else if (type < 27) {   // 10 - 26
      switch(mode) {
        case QRMode.NUMBER         : return 12;
        case QRMode.ALPHA_NUM      : return 11;
        case QRMode.BYTE_8BIT      : return 16;
        case QRMode.KANJI          : return 10;
      }
    } else if (type < 41) {  // 27 - 40
      switch(mode) {
        case QRMode.NUMBER         : return 14;
        case QRMode.ALPHA_NUM      : return 13;
        case QRMode.BYTE_8BIT      : return 16;
        case QRMode.KANJI          : return 12;
      }
    } else {
      throw new Error('type:' + type);
    }
};

function BitBuffer() {
  let _buffer = new Array();
  let _length = 0;
  let _this = {};

  _this.getBuffer = function() { return _buffer; };
  _this.getAt = function(index) {
    const bufIndex = index >>> 3;
    return ((_buffer[bufIndex] >>> (7 - index % 8)) & 1) == 1;
  };
  _this.put = function(num, length) {
    for (let i = 0; i < length; ++i) {
      _this.putBit(((num >>> (length - i - 1)) & 1) == 1);
    }
  };
  _this.num_bits = function() { return _length; };
  _this.putBit = function(bit) {
    const bufIndex = _length >>> 3;
    if (_buffer.length <= bufIndex) _buffer.push(0);
    if (bit) {
      _buffer[bufIndex] |= (0x80 >>> (_length % 8));
    }
    ++_length;
  };
  return _this;
};

function StringToBytes(s) {
  let bytes = new Uint8Array(s.length);
  for (let i = 0; i < s.length; ++i) bytes[i] = (s.charCodeAt(i) & 0xff);
  return bytes;
}

function ByteBuffer(data) {
  const _mode = QRMode.BYTE_8BIT;
  const _bytes = StringToBytes(data);
  let _this = {};

  _this.getMode = function() { return _mode; };
  _this.getLength = function(buffer) { return _bytes.length; };
  _this.WriteTo = function(buffer, typeNumber) {
    buffer.put(_mode, 4);
    buffer.put(_bytes.length, GetLengthInBits(_mode, typeNumber));
    for (let i = 0; i < _bytes.length; ++i) buffer.put(_bytes[i], 8);
  };
  return _this;
};

//---------------------------------------------------------------------

// QRMode
const QRMode = {
  NUMBER :     1 << 0,
  ALPHA_NUM :  1 << 1,
  BYTE_8BIT :  1 << 2,
  KANJI :      1 << 3
};
// QRErrorCorrectLevel
const QRErrorCorrectLevel = { L : 1, M : 0, Q : 3, H : 2 };

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
const QRPositionTable = [
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
function GetBCHDigit(data) {
  let digit = 0;
  for (; data != 0; ++digit) data >>>= 1;
  return digit;
};

function GetBCHTypeInfo(data) {
  let d = data << 10;
  while (GetBCHDigit(d) - GetBCHDigit(G15) >= 0) {
    d ^= (G15 << (GetBCHDigit(d) - GetBCHDigit(G15)));
  }
  return ((data << 10) | d) ^ G15_MASK;
};

function GetBCHTypeNumber(data) {
  let d = data << 12;
  while (GetBCHDigit(d) - GetBCHDigit(G18) >= 0) {
    d ^= (G18 << (GetBCHDigit(d) - GetBCHDigit(G18)));
  }
  return (data << 12) | d;
};

// QRMath
const QRMath = function() {
  const EXP_TABLE = new Array(256);
  const LOG_TABLE = new Array(256);
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


function qrPolynomial(num, shift) {
  if (typeof num.length == 'undefined') {
    throw new Error(num.length + '/' + shift);
  }
  const _num = function() {
    let offset = 0;
    while (offset < num.length && num[offset] == 0) ++offset;
    const _num = new Array(num.length - offset + shift);
    for (let i = 0; i < num.length - offset; ++i) _num[i] = num[i + offset];
    return _num;
  }();

  let _this = {};
  _this.getAt = function(index) { return _num[index]; };
  _this.getLength = function() { return _num.length; };
  _this.multiply = function(e) {
    let num = new Array(_this.getLength() + e.getLength() - 1);
    for (let i = 0; i < _this.getLength(); ++i) {
      for (let j = 0; j < e.getLength(); ++j) {
        num[i + j] ^= QRMath.gexp(QRMath.glog(_this.getAt(i)) + QRMath.glog(e.getAt(j)));
      }
    }
    return qrPolynomial(num, 0);
  };
  _this.mod = function(e) {
    if (_this.getLength() - e.getLength() < 0) return _this;
    const ratio = QRMath.glog(_this.getAt(0)) - QRMath.glog(e.getAt(0));
    let num = new Array(_this.getLength());
    for (let i = 0; i < _this.getLength(); ++i) num[i] = _this.getAt(i);
    for (let i = 0; i < e.getLength(); ++i) {
      num[i] ^= QRMath.gexp(QRMath.glog(e.getAt(i)) + ratio);
    }
    // recursive call
    return qrPolynomial(num, 0).mod(e);
  };
  return _this;
};

function GetPatternPosition(typeNumber) { return QRPositionTable[typeNumber - 1]; };
function GetMaskFunction(pattern_id) {
    switch (pattern_id) {
      case QRMaskPattern.PATTERN000 : return function(i, j) { return (i + j) % 2 == 0; };
      case QRMaskPattern.PATTERN001 : return function(i, j) { return i % 2 == 0; };
      case QRMaskPattern.PATTERN010 : return function(i, j) { return j % 3 == 0; };
      case QRMaskPattern.PATTERN011 : return function(i, j) { return (i + j) % 3 == 0; };
      case QRMaskPattern.PATTERN100 : return function(i, j) { return (Math.floor(i / 2) + Math.floor(j / 3)) % 2 == 0; };
      case QRMaskPattern.PATTERN101 : return function(i, j) { return (i * j) % 2 + (i * j) % 3 == 0; };
      case QRMaskPattern.PATTERN110 : return function(i, j) { return ( (i * j) % 2 + (i * j) % 3) % 2 == 0; };
      case QRMaskPattern.PATTERN111 : return function(i, j) { return ( (i * j) % 3 + (i + j) % 2) % 2 == 0; };
      default : throw new Error('bad maskPattern:' + pattern_id);
    }
};
function GetErrorCorrectPolynomial(errorCorrectLength) {
  let a = qrPolynomial([1], 0);
  for (let i = 0; i < errorCorrectLength; ++i) {
    a = a.multiply(qrPolynomial([1, QRMath.gexp(i)], 0));
  }
  return a;
};

function MakePayload(typeNumber, errorCorrectLevel, text) {
  const data = ByteBuffer(text);
  const blocks = QRRSBlock.GetRSBlocks(typeNumber, QRErrorCorrectLevel[errorCorrectLevel]);
  let buffer = BitBuffer();
  data.WriteTo(buffer, typeNumber);
  let total_bits = 0;
  for (const b of blocks) total_bits += b.dataCount;
  total_bits *= 8;
  if (buffer.num_bits() > total_bits) {
    throw new Error('code length overflow. (' + 
                    buffer.num_bits() + '>' + total_bits + ')');
  }
  if (buffer.num_bits() + 4 <= total_bits) buffer.put(0, 4);  // end code
  while (buffer.num_bits() % 8 != 0) buffer.putBit(false);    // padding
  while (true) {  // more padding
    if (buffer.num_bits() >= total_bits) break;
    buffer.put(PAD0, 8);
    if (buffer.num_bits() >= total_bits) break;
    buffer.put(PAD1, 8);
  }
  return CreateBytes(buffer, blocks);
}

//---------------------------------------------------------------------
// Scoring function for distortion

function GetScore(qrcode) {
  const moduleCount = qrcode._moduleCount;
  let score = 0;
  // LEVEL1
  let darkCount = 0;
  for (let y = 0; y < moduleCount; ++y) {
    for (let x = 0; x < moduleCount; ++x) {
      let sameCount = 0;
      const dark = qrcode.isDark(y, x);
      if (dark) ++darkCount;  // for LEVEL4
      for (let r = -1; r <= 1; ++r) {
        if (y + r < 0 || y + r >= moduleCount) continue;
        for (let c = -1; c <= 1; ++c) {
          if (x + c < 0 || x + c >= moduleCount) continue;
          if (r == 0 && c == 0) continue;
          if (dark == qrcode.isDark(y + r, x + c)) ++sameCount;
        }
      }
      if (sameCount > 5) score += (3 + sameCount - 5);
    }
  }
  // LEVEL2
  for (let y = 0; y < moduleCount - 1; ++y) {
    for (let x = 0; x < moduleCount - 1; ++x) {
      let count = 0;
      if (qrcode.isDark(y + 0, x + 0)) ++count;
      if (qrcode.isDark(y + 1, x + 0)) ++count;
      if (qrcode.isDark(y + 0, x + 1)) ++count;
      if (qrcode.isDark(y + 1, x + 1)) ++count;
      if (count == 0 || count == 4) score += 3;
    }
  }
  // LEVEL3
  for (let y = 0; y < moduleCount; ++y) {
    for (let x = 0; x < moduleCount - 6; ++x) {
      if (qrcode.isDark(y, x) && !qrcode.isDark(y, x + 1)
                              &&  qrcode.isDark(y, x + 2)
                              &&  qrcode.isDark(y, x + 3)
                              &&  qrcode.isDark(y, x + 4)
                              && !qrcode.isDark(y, x + 5)
                              &&  qrcode.isDark(y, x + 6)) {
        score += 40;
      }
    }
  }
  for (let x = 0; x < moduleCount; ++x) {
    for (let y = 0; y < moduleCount - 6; ++y) {
      if (qrcode.isDark(y, x) && !qrcode.isDark(y + 1, x)
                              &&  qrcode.isDark(y + 2, x)
                              &&  qrcode.isDark(y + 3, x)
                              &&  qrcode.isDark(y + 4, x)
                              && !qrcode.isDark(y + 5, x)
                              &&  qrcode.isDark(y + 6, x)) {
        score += 40;
      }
    }
  }
  // LEVEL4
  const ratio = Math.abs(20 * darkCount / moduleCount / moduleCount - 10);
  score += ratio * 10;
  return score;
};

//---------------------------------------------------------------------
// qrcode main object

class qrcode {
  constructor(typeNumber, errorCorrectLevel) {
    this._typeNumber = typeNumber;
    this._errorCorrectLevel = QRErrorCorrectLevel[errorCorrectLevel];
    this._moduleCount = typeNumber * 4 + 17;
    this._modules = null;
  };

  MakeData(data, test, pattern_id, onlyControl) {
    this._modules = new Array(this._moduleCount);
    for (let y = 0; y < this._moduleCount; ++y) {
      this._modules[y] = new Array(this._moduleCount);
      for (let x = 0; x < this._moduleCount; ++x) {
        this._modules[y][x] = null;
      }
    }
    this.SetupPositionProbePattern(3, 3);
    this.SetupPositionProbePattern(this._moduleCount - 4, 3);
    this.SetupPositionProbePattern(3, this._moduleCount - 4);
    this.SetupPositionAdjustPattern();
    this.SetupTimingPattern();
    this.SetupTypeInfo(test, pattern_id);
    this.SetupTypeNumber(this._typeNumber, test);      
    if (!onlyControl) this.MapData(data, pattern_id);
    return this._modules;
  }

  GetBestPattern(data) {
    let best_score = 0;
    let best_pattern_id = 0;
    for (let pattern_id = 0; pattern_id <= 7; ++pattern_id) {
      this.MakeData(data, true, pattern_id, false);
      const score = GetScore(this);
      // console.log("  try pattern: " + pattern_id + " score:" + score);
      if (pattern_id == 0 || score < best_score) {
        best_score = score;
        best_pattern_id = pattern_id;
      }
    }
    return best_pattern_id;
  }

  SetupPositionProbePattern(Y, X) {
    for (let y = -4; y <= 4; ++y) {
      if (Y + y < 0 || Y + y >= this._moduleCount) continue;
      for (let x = -4; x <= 4; ++x) {
        if (X + x < 0 || X + x >= this._moduleCount) continue;
        const v = Math.max(Math.abs(x), Math.abs(y));
        this._modules[Y + y][X + x] = (v != 2 && v != 4);
      }
    }
  }
  SetupPositionAdjustPattern() {
    const pos = GetPatternPosition(this._typeNumber);
    for (let j = 0; j < pos.length; ++j) {
      for (let i = 0; i < pos.length; ++i) {
        const X = pos[i], Y = pos[j];
        if (this._modules[Y][X] != null) continue;
        for (let y = -2; y <= 2; ++y) {
          for (let x = -2; x <= 2; ++x) {
            const mod = (Math.max(Math.abs(x), Math.abs(y)) != 1);
            this._modules[Y + y][X + x] = mod;
          }
        }
      }
    }
  }
  SetupTimingPattern() {
    for (let i = 8; i < this._moduleCount - 8; ++i) {
      if (this._modules[i][6] == null) this._modules[i][6] = (i % 2 == 0);
      if (this._modules[6][i] == null) this._modules[6][i] = (i % 2 == 0);
    }
  }
  SetupTypeNumber(typenumber, test) {
    if (typenumber < 7) return;
    const bits = GetBCHTypeNumber(typenumber);
    for (let i = 0; i < 18; ++i) {
      const Y = Math.floor(i / 3), X = (i % 3) + this._moduleCount - 8 - 3;
      const mod = (!test && ((bits >> i) & 1) == 1);
      this._modules[Y][X] = this._modules[X][Y] = mod;
    }
  }
  SetupTypeInfo(test, pattern_id) {
    const data = (this._errorCorrectLevel << 3) | pattern_id;
    const bits = GetBCHTypeInfo(data);
    for (let i = 0; i < 15; ++i) {
      const mod = (!test && ((bits >> i) & 1) == 1);
      // vertical
      if (i < 6) {
        this._modules[i][8] = mod;
      } else if (i < 8) {
        this._modules[i + 1][8] = mod;
      } else {
        this._modules[this._moduleCount - 15 + i][8] = mod;
      }
      // horizontal
      if (i < 8) {
        this._modules[8][this._moduleCount - i - 1] = mod;
      } else if (i < 9) {
        this._modules[8][15 - i - 1 + 1] = mod;
      } else {
        this._modules[8][15 - i - 1] = mod;
      }
    }
    // fixed module
    this._modules[this._moduleCount - 8][8] = (!test);
  }

  MapData(data, pattern_id) {
    const last = this._moduleCount - 1;
    let inc = -1;
    let row = last;
    let bitIndex = 7;
    let byteIndex = 0;
    let maskFunc = GetMaskFunction(pattern_id);
    for (let col = last; col > 0; col -= 2) {
      if (col == 6) col -= 1;
      while (true) {
        for (let c = 0; c < 2; ++c) {
          if (this._modules[row][col - c] == null) {
            let dark = false;
            if (byteIndex < data.length) {
              dark = (((data[byteIndex] >>> bitIndex) & 1) == 1);
            }
            const mask = maskFunc(row, col - c);
            if (mask) dark = !dark;
            this._modules[row][col - c] = dark;
            if (--bitIndex < 0) {
              ++byteIndex;
              bitIndex = 7;
            }
          }
        }
        row += inc;
        if (row < 0 || row > last) {
          row -= inc;
          inc = -inc;
          break;
        }
      }
    }
  }

  isDark(Y, X) { return this._modules[Y][X]; };
};
