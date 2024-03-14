import React, { Component, Fragment, RefObject } from 'react';
import tableCss from 'compDir/Theme/tableCss';
import { createStyles, Theme, withStyles } from '@material-ui/core/styles';
import MDIcon from '@mdi/react';
import { mdiDownload, mdiMenuUp, mdiMenuDown } from '@mdi/js';
import {
	Box, FormGroup, Grid, Typography, Checkbox, Radio, Paper, Card
} from '@material-ui/core';
import Spinner from 'compDir/Spinner';
import Modal from 'compDir/Modal';
import {
	IGraphqlPageInfo, ITask, ICategoryOption,
	IMedia, ITaskExportOpt, IHoliday, ITaskSearchOpt,
} from 'interfaceDir';
import { DatePicker } from 'compDir/DateTimePicker';
import { Paging, TaskStatus, ExportModes, TIME_ZONE } from 'constDir';
import Utils from 'servDir/utils';

import moment from 'moment-timezone';
moment.tz.setDefault(TIME_ZONE);

import ColorButton from 'rootDir/components/Button';

export interface IStateToProps {
	taskList: Array<ITask>
	largeCategories: Array<ICategoryOption>
	mediaList: Array<IMedia>
	holidayList: Array<IHoliday>
	pageInfo: IGraphqlPageInfo
	isLoading: boolean
	currentUser: any
}

export interface IDispatchToProps {
	initData(_comp: React.Component): any
}

interface IProps extends IDispatchToProps, IStateToProps {
	pauseInitData?: boolean
}

interface IState {
	firstLoading?: boolean
}
const useStyles = (theme: Theme) => createStyles({
	...tableCss,
	tableBody: {
		['& tr:nth-of-type(4n-1)']: {
			backgroundColor: theme.palette.action.hover
		}
	},
	modal: {
		margin: 0,
		position: 'absolute',
		top: 0
	},
});

class Page extends Component<IProps, IState> {

	// Set default properties's values
	public static defaultProps: IProps = {
	}

	// Set default state
	public state: IState = {
		firstLoading: true,
		scheduleId: 0,
		searchBarRef: React.createRef(),
	}

	componentDidMount() {
		// Load data for the first time load page --> trigger Only once-time 
		if (!this.props.pauseInitData) {
			this.props.initData(this);
		}
	}

	async componentDidUpdate(prevProps: IProps, prevState: IState, snapshot) {
		// init data on change page after clicked on menu --> Trigger multi-time
		if (this.props.pauseInitData != prevProps.pauseInitData) {
			this.props.initData(this);
		}
	}

	public render(): React.ReactNode {
		const { classes } = this.props;
		const { exportOptions, confirmBox } = this.state;

		if (this.state.firstLoading != false) {
			return <Spinner title="Loading..." />;
		}

		return (
			<Fragment>
				<Grid container spacing={4} direction='column' wrap='nowrap'>
					
				</Grid>
			</Fragment >
		);
	}
}
export default withStyles(useStyles)(Page);