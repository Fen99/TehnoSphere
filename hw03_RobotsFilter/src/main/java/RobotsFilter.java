import java.util.ArrayList;
import java.util.regex.Pattern;

public class RobotsFilter {
    private static final String StartPattern = "Disallow: ";
    private static final int StartPatternLen = StartPattern.length();

    public class BadFormatException extends Exception {
        public BadFormatException(String explanation) {
            super(explanation);
        }
    }

    private ArrayList<Pattern> patterns_ = new ArrayList<>();
    public RobotsFilter() {
    }

    public RobotsFilter(String robots_txt) throws BadFormatException {
        if (robots_txt.length() == 0) {
            return;
        }

        String[] rules = robots_txt.split("\n");
        for (String rule_str: rules) {
            if (!rule_str.startsWith(StartPattern)) {
                throw new BadFormatException("Rule should start with pattern '"+StartPattern+"'! Rule: "+rule_str+"; Robots.txt: "+robots_txt);
            }
            rule_str = rule_str.substring(StartPatternLen);

            String pattern_str = rule_str;
            if (rule_str.startsWith("/")) {
                pattern_str = "^\\Q"+pattern_str+"\\E.*";
            } else if (rule_str.startsWith("*")) {
                pattern_str = pattern_str.substring(1); // Remove *
                pattern_str = ".*\\Q"+pattern_str+"\\E.*";
            } else {
                if (pattern_str.endsWith("$")) {
                    pattern_str = ".*\\Q"+pattern_str+"\\E$";
                } else {
                    throw new BadFormatException("Unknown rule format: "+rule_str);
                }
            }
            if (pattern_str.endsWith("$\\E.*")) { // *<...>$ or /<...>$
                pattern_str = pattern_str.substring(0, pattern_str.length()-5)+"\\E$";
            }

            patterns_.add(Pattern.compile(pattern_str));
        }
    }

    public boolean IsAllowed(String url_str) throws BadFormatException {
        if (!url_str.startsWith("/")) {
            url_str = "/"+url_str;
        }

        for (Pattern p: patterns_) {
            if (p.matcher(url_str).matches()) {
                return false;
            }
        }
        return true;
    }
}
