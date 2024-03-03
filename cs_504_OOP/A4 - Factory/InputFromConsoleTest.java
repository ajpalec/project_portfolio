import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.FileNotFoundException;
import java.util.List;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

public class InputFromConsoleTest {

	List<String> list;
	ScannerWrapper scannerWrapper;
	SystemWrapper systemWrapper;
	InputFromConsole input;

	@BeforeEach
	public void setUp() throws FileNotFoundException {
		scannerWrapper = Mockito.mock(ScannerWrapper.class);
		systemWrapper = Mockito.mock(SystemWrapper.class);
		input = new InputFromConsole(scannerWrapper, systemWrapper);

	}

	@Test
	public void input_from_scanner() throws FileNotFoundException {
		Mockito.when(scannerWrapper.nextLine()).thenReturn("Descent of Man", "The Ascent of Man",
				"The Old Man and The Sea", "A Portrait of The Artist As a Young Man", "-1");
		list = input.read();
		Mockito.verify(systemWrapper).println("Please enter lines to add, then enter -1 to finish:");
		assertEquals("Descent of Man", list.get(0));
		assertEquals("The Ascent of Man", list.get(1));
		assertEquals("The Old Man and The Sea", list.get(2));
		assertEquals("A Portrait of The Artist As a Young Man", list.get(3));
	}

}
