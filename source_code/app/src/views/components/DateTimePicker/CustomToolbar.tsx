import React from 'react';
import PickerToolbar from '@material-ui/pickers/_shared/PickerToolbar';
import { makeStyles } from '@material-ui/core';
import { Grid } from '@material-ui/core';
import ColorButton from 'compDir/Button';

export const useStyles = makeStyles({
	toolbar: {
		padding: 10,
		height: 'auto',
		display: 'flex',
		flexDirection: 'column',
		justifyContent: 'center',
		alignItems: 'center',
	}
});

type IProps = {
}

const defaultProps: IProps = {
};

const CustomToolbar: React.FC<IProps> = (props) => {

	const { date, isLandscape, openView, setOpenView, title } = props;

	const handleChangeViewClick = (view) => (e) => {
		setOpenView(view);
	};

	const classes = useStyles();
	return (
		<PickerToolbar className={classes.toolbar} title={title} isLandscape={isLandscape}>
			{/* <Grid container justify='flex-start'>
				<Grid item>
					<ColorButton btnColor={'yellow'} btnContrast={600} size="small"
						startIcon={<MDIcon size={'18px'} path={mdiCalendarCheck} />}
						onClick={() => { }}>今日</ColorButton>
				</Grid>
			</Grid> */}
			<Grid container>
				<Grid item container alignItems='center' justify='center' spacing={1}>
					<Grid item >
						<ColorButton style={{ fontSize: 20, color: 'white', padding: 0, minWidth: 'auto' }}>
							({date.format('dddd')})
						</ColorButton>
					</Grid>
					<Grid item>
						<ColorButton style={{ fontSize: 20, color: 'white', padding: 0, minWidth: 'auto' }}
							onClick={handleChangeViewClick('year')}
							selected={openView === 'year'}
						>{date.format('YYYY年')}</ColorButton>
					</Grid>
					<Grid item>
						<ColorButton style={{ fontSize: 20, color: 'white', padding: 0, minWidth: 'auto' }}
							onClick={handleChangeViewClick('month')}
							selected={openView === 'month'}
						>{date.format('MMM')}</ColorButton>
					</Grid>

					<Grid item>
						<ColorButton style={{ fontSize: 20, color: 'white', padding: 0, minWidth: 'auto' }}
							onClick={handleChangeViewClick('date')}
							selected={openView === 'date'}
						>{date.format('Do')}</ColorButton>
					</Grid>
				</Grid>
			</Grid>
		</PickerToolbar>
	);
};

CustomToolbar.defaultProps = defaultProps;
export default CustomToolbar;