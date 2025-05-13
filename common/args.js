////////////////////////////////////////////////////////////////////////////////
// Parsing of URL params
//
//  defines: parse_arg(), parse_arg_str(), parse_arg_bool()

const args = new URLSearchParams(location.search);
const help = { V: new Array(), S: new Array(), B: new Array() };
function def_arg(v) { return v == undefined ? '?' : v; }
function parse_arg(arg_value, default_value, min, max) {
  let v = args.has(arg_value) ? parseFloat(args.get(arg_value)) : default_value;
  if (min != undefined) v = Math.max(v, min);
  if (max != undefined) v = Math.min(v, max);
  help.V.push(`${arg_value}`.padEnd(32) +
              `${def_arg(min)} <= ${def_arg(default_value)} <= ${def_arg(max)}`);
  return v;
}

function parse_arg_str(arg_value, default_value) {
  help.S.push(`${arg_value}`.padEnd(32) + `[${def_arg(default_value)}]`);
  return args.has(arg_value) ? args.get(arg_value) : default_value;
}

function parse_arg_bool(arg_value) {
  help.B.push(`${arg_value}`.padEnd(32));
  return (args.has(arg_value) == true);
}

function arg_help() {
  let str = `URL params:\n`;
  str += '\n== values ==\n';
  for (const C of help.V) str += ` - ${C}\n`;
  str += '\n== strings ==\n';
  for (const C of help.S) str += ` - ${C}\n`;
  str += '\n== bool flags ==\n';
  for (const C of help.B) str += ` - ${C}\n`;
  console.log(str);
}
