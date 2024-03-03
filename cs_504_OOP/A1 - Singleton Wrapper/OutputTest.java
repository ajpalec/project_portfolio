import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.InOrder;
import org.mockito.Mockito;

class OutputTest {

	Output output;
	SystemWrapper systemWrapper;

	@BeforeEach
	void setUp() {
		output = new Output();
		systemWrapper = Mockito.mock(SystemWrapper.class);
	}

	@Test
	void output_lines_to_file() {
		List<String> lines = new ArrayList<>();
		lines.add("hi bye");
		lines.add("hi bye foo");
		output.write(lines, systemWrapper);
		InOrder inOrder = Mockito.inOrder(systemWrapper);
		inOrder.verify(systemWrapper).println("hi bye");
		inOrder.verify(systemWrapper).println("hi bye foo");
		inOrder.verifyNoMoreInteractions();
	}

}
