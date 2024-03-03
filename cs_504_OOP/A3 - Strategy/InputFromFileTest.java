import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class InputFromFileTest {

	List<String> list;
	InputFromFile input;

	@BeforeEach
	public void setUp() {
		input = new InputFromFile();
	}

	@Test
	public void input_from_scanner() throws IOException {
		List<String> lines = new ArrayList<>();
		lines.add("Descent of Man");
		lines.add("The Ascent of Man");
		lines.add("The Old Man and The Sea");
		lines.add("A Portrait of The Artist As a Young Man");
		FileWriter fileWriter = new FileWriter("kwic.txt");
		for (String line : lines) {
			fileWriter.write(line);
			fileWriter.write("\n");
		}
		fileWriter.close();
		list = input.read();
		assertSame(4, list.size());
		assertEquals("Descent of Man", list.get(0));
		assertEquals("The Ascent of Man", list.get(1));
		assertEquals("The Old Man and The Sea", list.get(2));
		assertEquals("A Portrait of The Artist As a Young Man", list.get(3));
	}

}
