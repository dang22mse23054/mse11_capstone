import { createMuiTheme } from '@material-ui/core/styles';
const theme = createMuiTheme({
	palette: {
		primary: {
			main: '#6AB187',
			dark: '#BC5100',
			contrastText: '#FAFAFA'
		},
		secondary: {
			main: '#31473A',
			light: '#62727B',
			dark: '#102027',
			contrastText: '#FAFAFA'
		},
		success: {
			main: '#4caf50',
			light: '#8BC34A'
		},
		error: {
			main: '#f44336',
			light: '#FF80AB'
		},
		text: {
			primary: 'rgba(0, 0, 0, 0.87)',
			secondary: 'rgba(0, 0, 0, 0.60)',
			disabled: 'rgba(0, 0, 0, 0.38)',
			hint: 'rgba(0, 0, 0, 0.38)'
		},
		action: {
			selected: 'rgba(0, 0, 0, 0.16)',
			// active: '#F57F17'
		},
	},
});

export default theme;