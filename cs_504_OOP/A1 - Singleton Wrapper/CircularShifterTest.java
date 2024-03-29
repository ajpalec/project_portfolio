import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class CircularShifterTest {

	CircularShifter circularShifter;

	@BeforeEach
	void setUp() {
		circularShifter = new CircularShifter();
	}

	@Test
	void shift_2_lines_completely() {
		List<String> lines = new ArrayList<>();
		lines.add("hi bye foo");
		lines.add("hi bye");
		List<String> actual = circularShifter.shiftLines(lines);
		assertSame(5, actual.size());
		assertTrue(actual.contains("bye hi"));
		assertTrue(actual.contains("hi bye"));
		assertTrue(actual.contains("foo hi bye"));
		assertTrue(actual.contains("bye foo hi"));
		assertTrue(actual.contains("hi bye foo"));
	}

}
