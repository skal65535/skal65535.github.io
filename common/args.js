////////////////////////////////////////////////////////////////////////////////
// Parsing of URL params
//
//  defines: parse_arg(), parse_arg_str(), parse_arg_bool()

const args = new URLSearchParams(location.search);
function parse_arg(arg_value, default_value, min, max) {
  let v = args.has(arg_value) ? parseFloat(args.get(arg_value)) : default_value;
  if (min != undefined) v = Math.max(v, min);
  if (max != undefined) v = Math.min(v, max);
  return v;
}
function parse_arg_str(arg_value, default_value) {
  return args.has(arg_value) ? args.get(arg_value) : default_value;
}
function parse_arg_bool(arg_value) { return (args.has(arg_value) == true); }
