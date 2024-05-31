import React, { FC, forwardRef } from 'react';
import { Box, LinearProgress, Typography } from '@material-ui/core';

interface IProps {
	value: number
	chunks: number
}

const CustomLinearProgress: FC<IProps> = forwardRef((props: IProps, ref) => {
	const value = Math.round((props.value / props.chunks + Number.EPSILON) * 100);
	const buffer = Math.round(((props.value) / props.chunks + Number.EPSILON) * 100) + 5;
	return (
		<Box display="flex" alignItems="center">
			<Box width="100%" mr={1}>
				<LinearProgress variant="buffer" {...props} value={value} valueBuffer={buffer}/>
			</Box>
			<Box minWidth={35}>
				<Typography variant="body2" color="textSecondary">
					{`${value}%`}</Typography>
			</Box>
		</Box>
	);
});

export default CustomLinearProgress;