import static org.junit.Assert.*;
import org.junit.Test;

public class RobotsFilterTest {
    @Test
    public void testSimpleCase() throws RobotsFilter.BadFormatException {
        RobotsFilter filter = new RobotsFilter("Disallow: /users");

        assertTrue(filter.IsAllowed("/company/about.html"));

        assertFalse(filter.IsAllowed("/users/jan"));
        assertFalse(filter.IsAllowed("/users/"));
        assertFalse(filter.IsAllowed("/users"));

        assertTrue("should be allowed since in the middle", filter.IsAllowed("/another/prefix/users/about.html"));
        assertTrue("should be allowed since at the end", filter.IsAllowed("/another/prefix/users"));
    }

    @Test
    public void testEmptyCase() throws RobotsFilter.BadFormatException {
        RobotsFilter filter = new RobotsFilter();

        assertTrue(filter.IsAllowed("/company/about.html"));
        assertTrue(filter.IsAllowed("/company/second.html"));
        assertTrue(filter.IsAllowed("any_url"));
    }

    @Test
    public void testEmptyStringCase() throws RobotsFilter.BadFormatException {
        // that's different from testEmptyCase() since we
        // explicitly pass empty robots_txt rules
        RobotsFilter filter = new RobotsFilter("");

        assertTrue(filter.IsAllowed("/company/about.html"));
        assertTrue(filter.IsAllowed("/company/second.html"));
        assertTrue(filter.IsAllowed("any_url"));
    }

    @Test
    public void testRuleEscaping() throws RobotsFilter.BadFormatException {
        // we have to escape special characters in rules (like ".")
        RobotsFilter filter = new RobotsFilter("Disallow: *.php$");

        assertFalse(filter.IsAllowed("file.php"));
        assertTrue("sphp != .php", filter.IsAllowed("file.sphp"));
    }

    @Test(expected = RobotsFilter.BadFormatException.class)
    public void testBadFormatException() throws RobotsFilter.BadFormatException {
        RobotsFilter filter = new RobotsFilter("Allowed: /users");
    }

    @Test
    public void testAllCases() throws RobotsFilter.BadFormatException {
        String rules = "Disallow: /users\n" +
                "Disallow: *.php$\n" +
                "Disallow: */cgi-bin/\n" +
                "Disallow: /very/secret.page.html$\n";

        RobotsFilter filter = new RobotsFilter(rules);

        assertFalse(filter.IsAllowed("/users/jan"));
        assertTrue("should be allowed since in the middle", filter.IsAllowed("/subdir2/users/about.html"));

        assertFalse(filter.IsAllowed("/info.php"));
        assertTrue("we disallowed only the endler", filter.IsAllowed("/info.php?user=123"));
        assertTrue(filter.IsAllowed("/info.pl"));

        assertFalse(filter.IsAllowed("/forum/cgi-bin/send?user=123"));
        assertFalse(filter.IsAllowed("/forum/cgi-bin/"));
        assertFalse(filter.IsAllowed("/cgi-bin/"));
        assertTrue(filter.IsAllowed("/scgi-bin/"));

        assertFalse(filter.IsAllowed("/very/secret.page.html"));
        assertTrue("we disallowed only the whole match", filter.IsAllowed("/the/very/secret.page.html"));
        assertTrue("we disallowed only the whole match", filter.IsAllowed("/very/secret.page.html?blah"));
        assertTrue("we disallowed only the whole match", filter.IsAllowed("/the/very/secret.page.html?blah"));
    }

    static public void main(String[] args) throws Exception {
        RobotsFilterTest test = new RobotsFilterTest();
        try {
            test.testBadFormatException();
        } catch (RobotsFilter.BadFormatException exc) {
            System.out.println("BadFormat test passed!");

            test.testEmptyStringCase();
            test.testSimpleCase();
            test.testRuleEscaping();
            test.testAllCases();
            test.testEmptyCase();

            System.out.println("All passed!");
            return;
        }

        throw new Exception("Bad format failed!");
    }
}